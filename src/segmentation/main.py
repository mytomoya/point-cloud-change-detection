"""Panoptic segmentation module."""

import collections
import json
import pathlib
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm

# Detectron2
from detectron2 import checkpoint, model_zoo, modeling
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling.meta_arch.panoptic_fpn import PanopticFPN

from src import utils
from src.segmentation import helper
from src.segmentation.type import (
    InfoType,
    InputType,
    StuffSegmentType,
    ThingSegmentType,
)

# Suppress "UserWarning: torch.meshgrid"
warnings.simplefilter("ignore", UserWarning)


class PanopticSegmentation:
    """Panoptic segmentation class."""

    def __init__(self, dataset_path: pathlib.Path, batch_size: int = 2):
        # Set path
        self.dataset_path = dataset_path
        self.label_directory = utils.get_label_path(dataset_path)
        self.rgb_directory = utils.get_rgb_path(dataset_path)
        self.json_path = utils.get_unpair_path(dataset_path)

        self.batch_size = batch_size

        # Create `Label` directory if it does not exist
        self.label_directory.mkdir(exist_ok=True)

        self.config = self.setup_config()

        # Resize setting
        self.aug = T.ResizeShortestEdge(
            [self.config.INPUT.MIN_SIZE_TEST, self.config.INPUT.MIN_SIZE_TEST],
            self.config.INPUT.MAX_SIZE_TEST,
        )

        # Dataset used for training
        dataset = self.config.DATASETS.TRAIN[0]

        # Get its metadata
        metadata = MetadataCatalog.get(dataset)

        # Thing/stuff classes in the dataset
        self.stuff_classes: list[str] = metadata.stuff_classes
        self.thing_classes: list[str] = metadata.thing_classes

        # Predictor
        self.model: PanopticFPN = modeling.build_model(self.config)
        checkpoint.DetectionCheckpointer(self.model).load(self.config.MODEL.WEIGHTS)
        self.model.eval()

        self.instance_info: dict[str, InfoType] = collections.defaultdict(
            lambda: {"count": 0, "unpair_list": []}
        )

    def setup_config(self) -> CfgNode:
        """Setup the configuration for the panoptic segmentation model.

        Returns
        -------
        config : CfgNode
            Configuration for the panoptic segmentation model.
        """
        config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        config_path = model_zoo.get_config_file(config_file)
        model_url = model_zoo.get_checkpoint_url(config_file)

        config = get_cfg()
        config.merge_from_file(config_path)

        config.MODEL.DEVICE = "cpu"
        config.MODEL.WEIGHTS = model_url

        return config

    def run(self, step: int = 1):
        """Run the panoptic segmentation model.

        Notes
        -----
        The result of the panoptic segmentation for each image looks like this:
        ```
        {
            "sem_seg": torch.Tensor,
            "instances": Instances,
            "panoptic_seg": ("pred", "segments_info")
        }
        ```

        `pred`: torch.Tensor
            Tensor of shape `(H, W)` giving the label of the segment for each pixel.
        `segments_info`: list[dict[str, Any]]
            Describe each segment in `panoptic_seg`. Each dict contains keys `id`, `category_id`,
            `isthing`. If `isthing` is `True`, `score` and `instance_id` are also available.
            Otherwise, `area` is available.

        For more information, see
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
        """
        n_frames = len(list(self.rgb_directory.iterdir()))

        image_batch: list[InputType] = []
        image_indices: list[int] = []

        for i in tqdm.tqdm(range(n_frames), desc="[Panoptic Segmentation]"):
            if i % step != 0:
                continue

            image_path = self.rgb_directory / f"rgb_{i}.png"
            if not image_path.exists():
                continue
            image = cv2.imread(image_path.as_posix())  # pylint: disable=no-member

            # Resize
            height, width = image.shape[:2]
            resized_image = self.aug.get_transform(image).apply_image(image)
            resized_image = resized_image.astype(np.float32)
            resized_image = resized_image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            torch_image = torch.as_tensor(resized_image)  # pylint: disable=no-member

            image_batch.append({"image": torch_image, "height": height, "width": width})
            image_indices.append(i)

            if len(image_batch) != self.batch_size and i != n_frames - 1:
                continue

            with torch.no_grad():
                result = self.model(image_batch)

            for j in range(len(image_batch)):
                index = image_indices[j]
                panoptic_seg, segments_info = result[j]["panoptic_seg"]
                labels = self.update_instance_info(panoptic_seg, segments_info)

                # Save the labels
                save_path = self.label_directory / f"label_{index}"
                np.save(save_path.as_posix(), labels)

            image_batch = []
            image_indices = []

        # Save the list of labels that should not be merged
        unpair_set = helper.get_unpair_set(self.instance_info)
        with open(self.json_path, mode="w", encoding="utf-8") as json_file:
            json.dump(tuple(unpair_set), json_file, indent=4)

    def update_instance_info(
        self,
        panoptic_seg: torch.Tensor,
        segments_info: list[StuffSegmentType | ThingSegmentType],
    ) -> np.ndarray:
        """Update the instance information.

        Parameters
        ----------
        panoptic_seg : torch.Tensor
            Tensor of shape `(H, W)` giving the label of the segment for each pixel.
        segments_info : list[StuffSegmentType | ThingSegmentType]
            Describe each segment in `panoptic_seg`. Each dict contains keys `id`, `category_id`,
            `isthing`. For segments with `isthing` being `True`, `instance_id` must be available.

        Returns
        -------
        labels : np.ndarray
            The instance label of each pixel. The shape is `(H, W)`.
        """
        segment_id_to_instance_label: dict[int, str] = collections.defaultdict(str)
        instance_list: dict[str, list[str]] = collections.defaultdict(list)

        for segment in tqdm.tqdm(segments_info, desc="[Segment]", leave=False):
            segment_id = segment["id"]
            category_id = segment["category_id"]

            object_label: str = (
                self.thing_classes[category_id]
                if segment["isthing"]
                else self.stuff_classes[category_id]
            )

            # Distinguish between instances
            if "instance_id" in segment:
                count = self.instance_info[object_label]["count"]
                instance_label = f"{object_label}_{count}"
                segment_id_to_instance_label[segment_id] = instance_label

                instance_list[object_label].append(instance_label)
                self.instance_info[object_label]["count"] += 1
            else:
                segment_id_to_instance_label[segment_id] = object_label

        for object_label, instance_labels in instance_list.items():
            self.instance_info[object_label]["unpair_list"].append(instance_labels)

        def get_instance_label(label_id: int) -> str:
            return segment_id_to_instance_label[label_id]

        panoptic_seg_numpy = panoptic_seg.detach().cpu().numpy()
        labels = np.frompyfunc(get_instance_label, 1, 1)(panoptic_seg_numpy)

        return labels


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Panoptic Segmentation")
    parser.add_argument("-p", "--path", help="Path to the dataset", type=str)

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path

    if path is None:
        # Use current directory if no path is provided
        data_path = pathlib.Path()
        data_path = pathlib.Path("../../dataset/raw/1/Before")
    else:
        data_path = pathlib.Path(path)

    panoptic_segmentation = PanopticSegmentation(dataset_path=data_path)
    panoptic_segmentation.run()
