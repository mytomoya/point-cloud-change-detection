"""Module for extracting point clouds from the processed dataset."""

import pathlib
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import tqdm

from src import Parameter, utils
from src.type import AnnotationItemType

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class Extraction:
    """Extracts point clouds from the processed dataset."""

    def __init__(
        self,
        root: pathlib.Path,
        used_labels: set[str] | None = None,
        min_points: int = Parameter.min_points,
    ):
        """
        Parameters
        ----------
        root pathlib.Path
            Path to the processed dataset that contains `Before` and `After` directories.
        used_labels : set[str] | None, default None
            Set of labels that are to be used. Labels not in this set are ignored.
        min_points : int, default Parameter.min_points
            Minimum number of points in a point cloud. If point cloud extracted from extracted has
            less points than this value, it is discarded.
        """

        self.before_path = root / "Before"
        self.after_path = root / "After"
        self.annotation_path = root / "annotation.json"
        self.json_data = utils.annotation.load(self.annotation_path)

        # Load point cloud and labels
        (
            self.point_cloud_before,
            self.labels_before,
        ) = self.load_point_cloud_and_label(self.before_path)
        (
            self.point_cloud_after,
            self.labels_after,
        ) = self.load_point_cloud_and_label(self.after_path)

        self.points_before = np.array(self.point_cloud_before.points)
        self.colors_before = np.array(self.point_cloud_before.colors)
        self.points_after = np.array(self.point_cloud_after.points)
        self.colors_after = np.array(self.point_cloud_after.colors)

        self.used_labels = (
            used_labels if used_labels is not None else Parameter.used_labels
        )
        self.min_points = min_points

    def load_point_cloud_and_label(
        self, dataset_path: pathlib.Path
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Loads the point cloud and label from the processed dataset.

        Parameters
        ----------
        dataset_path : pathlib.Path
            Path to the point cloud and label.

        Returns
        -------
        point_cloud : o3d.geometry.PointCloud
            Point cloud.
        label : np.ndarray
            Label.
        """

        point_cloud_path = utils.get_point_cloud_path(dataset_path, kind="registered")
        point_cloud = o3d.io.read_point_cloud(point_cloud_path.as_posix())
        label = np.load(dataset_path / "merged_label.npy", allow_pickle=True)

        return point_cloud, label

    def extract_with_label(
        self, instance_label: str
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """Extracts points with the specified label from `self.point_cloud_before`.

        Parameters
        ----------
        instance_label : str
            Label of points that are to be extracted.

        Returns
        -------
        point_cloud_before : o3d.geometry.PointCloud
            Extracted point cloud.
        labels : np.ndarray
            List of labels of each point.
        indices : np.ndarray
            List of indices of each point in the original point cloud.
        """

        indices = np.where(self.labels_before == instance_label)[0]

        labels = self.labels_before[self.labels_before == instance_label].copy()
        new_points = self.points_before[self.labels_before == instance_label]
        new_colors = self.colors_before[self.labels_before == instance_label]

        point_cloud_before = utils.create_point_cloud(new_points, new_colors)

        return point_cloud_before, labels, indices

    def extract_in_range(
        self, minimum: list[float], maximum: list[float]
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
        """Extracts points in the specified range from `self.point_cloud_after`.

        Parameters
        ----------
        minimum : List[float]
            Lower bound.
        maximum : List[float]
            Upper bound.

        Returns
        -------
        point_cloud_before : o3d.geometry.PointCloud
            Extracted point cloud.
        labels : np.ndarray
            List of labels of each point.
        indices : np.ndarray
            List of indices of each point in the original point cloud.
        """

        points = self.points_after.copy()
        colors = self.colors_after.copy()
        labels = self.labels_after.copy()

        condition = np.all(minimum <= points, axis=1) & np.all(
            points <= maximum, axis=1
        )
        indices = np.where(condition)[0]

        new_points = points[condition]
        new_colors = colors[condition]
        new_labels = labels[condition]

        point_cloud_after = utils.create_point_cloud(new_points, new_colors)

        return point_cloud_after, new_labels, indices

    def extract(self):
        """Extracts point clouds based on labels and ranges."""

        offset: int = 0
        with tqdm.tqdm(np.unique(self.labels_before)) as pbar:
            for instance_label in pbar:
                pbar.set_description(f"[Extracting {instance_label}]")
                object_label = utils.get_object_label(instance_label)

                if object_label not in self.used_labels:
                    continue

                (
                    point_cloud_before,
                    labels_before,
                    indices_before,
                ) = self.extract_with_label(instance_label)
                minimum, maximum = utils.pcd.get_boundary(point_cloud_before)

                if len(point_cloud_before.points) < self.min_points:
                    continue
                (
                    point_cloud_after,
                    labels_after,
                    indices_after,
                ) = self.extract_in_range(minimum=minimum, maximum=maximum)

                # Update annotation file
                item = AnnotationItemType(id=offset, object=object_label, change="")
                self.json_data = utils.annotation.add(self.json_data, item)
                utils.annotation.save(self.annotation_path, self.json_data)

                # Save point clouds
                self.save(
                    point_cloud_before,
                    labels_before,
                    indices_before,
                    self.before_path,
                    offset,
                )
                self.save(
                    point_cloud_after,
                    labels_after,
                    indices_after,
                    self.after_path,
                    offset,
                )

                offset += 1

    def save(
        self,
        point_cloud: o3d.geometry.PointCloud,
        labels: np.ndarray,
        indices: np.ndarray,
        dataset_path: pathlib.Path,
        index: int,
    ):
        """Saves the point clouds, labels, and indices.

        Parameters
        ----------
        point_cloud : o3d.geometry.PointCloud
            Point cloud.
        labels : np.ndarray
            List of labels of each point.
        indices : np.ndarray
            List of indices of each point in the original point cloud.
        dataset_path : pathlib.Path
            Path to the dataset.
        index : int
            Point cloud number.
        """

        ply_path = dataset_path / "PLY" / f"{index}.ply"
        label_path = dataset_path / "Label" / f"{index}"
        index_path = dataset_path / "Index" / f"{index}.npy"

        # Create directories if they do not exist
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save point clouds, labels, and indices
        o3d.io.write_point_cloud(
            ply_path.as_posix(),
            point_cloud,
            write_ascii=True,
            compressed=True,
            print_progress=True,
        )
        np.save(label_path.as_posix(), labels)
        np.save(index_path.as_posix(), indices)


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Panoptic Segmentation")
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the dataset that contains `Before` and `After` directories.",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Labels to be used.",
        nargs="*",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--min",
        help="Minimum number of points.",
        type=int,
    )

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path
    labels = set(args.labels) if args.labels is not None else Parameter.used_labels

    if path is None:
        root = pathlib.Path("../../dataset/processed/1")
    else:
        root = pathlib.Path(path)

    extraction = Extraction(root, used_labels=labels, min_points=args.min)
    extraction.extract()
