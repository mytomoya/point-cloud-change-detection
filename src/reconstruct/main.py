"""
Module for point cloud reconstruction
"""

import pathlib
from argparse import ArgumentParser
from typing import Literal

import numpy as np
import tqdm

from src import Parameter, utils
from src.reconstruct import constant
from src.reconstruct.constant import Confidence


class ReconstructPointCloud:
    """Class for point cloud reconstruction."""

    def __init__(
        self,
        root: pathlib.Path,
        num_samples_per_frame: int = Parameter.num_samples_per_frame,
        extension: Literal["jpg", "png"] = "png",
        step: int = Parameter.reconstruction_step,
    ):
        self.rgb_directory = utils.get_rgb_path(root)
        self.label_directory = utils.get_label_path(root)

        self.root = root
        self.num_samples_per_frame = num_samples_per_frame
        self.extension: Literal["jpg", "png"] = extension
        self.step = step

    def get_world_coordinate(
        self,
        y: np.ndarray,  # pylint: disable=invalid-name
        x: np.ndarray,  # pylint: disable=invalid-name
        depth_map: np.ndarray,
        intrinsic_inversed: np.ndarray,
        view_matrix_inversed: np.ndarray,
    ) -> np.ndarray:
        """Get the world coordinate of the given pixels.

        Parameters
        ----------
        y : np.ndarray
            y coordinate of the pixels.
        x : np.ndarray
            x coordinate of the pixels.
        depth_map : np.ndarray
            Depth map of shape (height, width).
        intrinsic_inversed : np.ndarray
            Inverse of the intrinsic matrix of shape (3, 3).
        view_matrix_inversed : np.ndarray
            Inverse of the view matrix of shape (4, 4).

        Returns
        -------
        world_coordinate : np.ndarray
            World coordinate of the given pixels.
        """
        depth = depth_map[y, x]

        position_in_camera_space = (
            intrinsic_inversed @ np.array([x, y, np.ones_like(x)]) * depth
        )

        corrected_position_in_camera_space = (
            constant.flipYZ
            @ constant.device_transform_matrix
            @ np.vstack(
                (
                    position_in_camera_space,
                    np.ones_like(position_in_camera_space[1]),
                )
            )
        )

        position_in_world_space = (
            view_matrix_inversed @ corrected_position_in_camera_space
        )

        position_in_world_space /= position_in_world_space[-1]

        return position_in_world_space[:3]

    def reconstruct(
        self, number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Reconstructs the point cloud from the 'number'-th RGB-D image. If the
        image does not exist, returns None.

        Parameters
        ----------
        number : int
            Image id.

        Returns
        -------
        points : np.ndarray
            Points of the point cloud of shape (3, None).
        color : np.ndarray
            Color of the point cloud of shape (3, None).
        label : np.ndarray
            Label of the point cloud of shape (3, None).
        """
        np.random.seed(0)

        if number % self.step != 0:
            return None
        loaded_images = utils.load_images(number, self.root, self.extension)
        # Check if the image has successfully been loaded
        if loaded_images is None:
            return None
        image, depth_map, confidence_map = loaded_images

        # Parameters
        depth_path = utils.get_depth_path(self.root, number)
        intrinsic, view_matrix = utils.get_parameters(depth_path.as_posix())

        intrinsic_inversed = np.linalg.inv(intrinsic)
        view_matrix_inversed = np.linalg.inv(view_matrix)

        y, x = np.where(confidence_map == Confidence.HIGH)
        n_samples = len(y)
        if n_samples > self.num_samples_per_frame:
            sampled = np.random.choice(np.arange(n_samples), self.num_samples_per_frame)
            y, x = y[sampled], x[sampled]

        points = self.get_world_coordinate(
            y,
            x,
            depth_map,
            intrinsic_inversed,
            view_matrix_inversed,
        )

        points = np.transpose(points, (1, 0))

        color = image[y, x].astype(np.float32)
        color /= 255

        # Load labels
        label_path = utils.get_label_path(self.root, number)
        if not label_path.exists():
            return None
        label = np.load(label_path.as_posix(), allow_pickle=True)
        label = np.rot90(label)

        # Sample labels
        label = label[y, x]

        # Reshape to (None, 3)
        label = label.reshape(-1, 1)
        label = np.repeat(label, repeats=3, axis=1)

        return points, color, label

    def run(self):
        """Reconstructs the point cloud from the RGB-D images. The reconstructed point cloud and
        the labels are saved as a .ply file and a .npy file, respectively.
        """
        n_frames: int = len(list(self.rgb_directory.iterdir()))
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for i in tqdm.tqdm(range(n_frames), desc="[Restore Point Cloud]"):
            result = self.reconstruct(i)
            if result is None:
                continue
            results.append(result)

        np_results = np.array(results)

        point_cloud = np_results[:, 0].reshape(-1, 3)
        colors = np_results[:, 1].reshape(-1, 3)
        labels = np_results[:, 2, :, 0].reshape(-1)

        utils.save_point_cloud(point_cloud, colors, self.root)
        save_path = self.root / "label"
        np.save(save_path.as_posix(), labels)


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Reconstruct point cloud from the dataset.")
    parser.add_argument("-p", "--path", help="Path to the dataset", type=str)
    parser.add_argument(
        "-n",
        "--num_samples",
        help="Number of samples per frame",
        type=int,
        default=5_000,
    )
    parser.add_argument("-s", "--step", help="Step size", type=int, default=1)
    parser.add_argument("-e", "--extension", help="Extension", type=str, default="png")

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path

    if path is None:
        current_directory = pathlib.Path("../../dataset/raw/1/Before")
    else:
        current_directory = pathlib.Path(path)

    reconstruct = ReconstructPointCloud(
        current_directory, args.num_samples, args.extension, args.step
    )
    reconstruct.run()
