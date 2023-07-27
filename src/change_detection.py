import pathlib
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import tqdm
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

from src import Parameter, utils
from src.type import ChangeType

# Suppress warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class ChangeDetection:
    """Change detection class."""

    def __init__(self, root: pathlib.Path, number: int):
        """Initializes the class.

        Parameters
        ----------
        root : pathlib.Path
            Path to the dataset. The dataset has to contain `Before` and `After` directories.
        number : int
            ID of the point cloud.
        """
        self.root = root
        self.number = number

        (
            self.point_cloud_before,
            self.labels_before,
        ) = utils.load_point_cloud_and_label(root / "Before", number=number)
        (
            self.point_cloud_after,
            self.labels_after,
        ) = utils.load_point_cloud_and_label(root / "After", number=number)

        # Convert instance label to object label
        instance_to_object_label = np.frompyfunc(
            lambda instance_label: instance_label.split("_")[0], 1, 1
        )
        self.temp = self.labels_after.copy()
        self.labels_before = instance_to_object_label(self.labels_before)
        self.labels_after = instance_to_object_label(self.labels_after)

        self.points_before = np.array(self.point_cloud_before.points)
        self.points_after = np.array(self.point_cloud_after.points)

        self.label = np.unique(self.labels_before)

    def check_number_of_points(self, n_neighbors: int) -> bool:
        """Checks if the number of points in the point cloud at t is less than
        the number of neighbors.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors.

        Returns
        -------
        result : bool
            `True` if the number of points in the point cloud at t is less than
            the number of neighbors, `False` otherwise.
        """
        return len(self.points_after) < n_neighbors

    def run(self, n_neighbors: int) -> float:
        """Runs the change detection.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to consider.

        Returns
        -------
        result : float
            Ratio of the number of matches to the number of checks.
        """
        if self.check_number_of_points(n_neighbors):
            return 0.0

        nn = NearestNeighbors()
        nn.fit(self.points_after)

        _, indices = nn.kneighbors(self.points_before, n_neighbors=n_neighbors)

        match_flag = self.labels_after[indices] == self.label
        n_matches = np.sum(match_flag)
        n_checks = n_neighbors * len(self.points_before)

        return n_matches / n_checks

    def color(
        self,
        color_before,
        color_after,
        prediction: ChangeType,
        label: ChangeType,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Color the point cloud.
        - Cyan: True positive. The point cloud is removed and the prediction is correct.
        - Green: False positive. The point cloud is not removed but the prediction is incorrect.
        - Magenta: False negative (miss). The point cloud is removed but the prediction is incorrect.
        - Red: True negative. The point cloud is not removed and the prediction is correct.

        Parameters
        ----------
        color_before : np.ndarray
            Color of the point cloud at t.
        color_after : np.ndarray
            Color of the point cloud at t + 1.
        prediction : ChangeType
            Prediction of the change detection.
        label : ChangeType
            Ground truth.

        Returns
        -------
        color_before : np.ndarray
            Updated color of the point cloud at t.
        color_after : np.ndarray
            Updated color of the point cloud at t + 1.
        """
        if prediction == ChangeType.REMOVE:
            color = utils.CYAN if label == prediction else utils.GREEN
        else:
            color = utils.MAGENTA if label == prediction else utils.RED

        path = self.root / "Before" / "Index" / f"{self.number}.npy"
        indices = np.load(path.as_posix(), allow_pickle=True)
        color_before[indices] = color

        path = self.root / "After" / "Index" / f"{self.number}.npy"
        indices = np.load(path.as_posix(), allow_pickle=True)
        color_after[indices] = color

        return color_before, color_after


def main(root: pathlib.Path, n_neighbors: int, threshold: float):
    """Main function. Runs the change detection and saves the colored point clouds and confusion
    matrix as a .ply file and a .svg file, respectively.

    Parameters
    ----------
    root : pathlib.Path
        Path to the dataset. The dataset has to contain `Before` and `After` directories.
    n_neighbors : int
        Number of neighbors to consider.
    threshold : float
        Threshold of match rate.
    """
    y_true = utils.annotation.get_ground_truth(root)
    y_prediction: list[ChangeType] = []
    number = len(y_true)

    all_colors_before: np.ndarray | None = None
    all_colors_after: np.ndarray | None = None
    all_points_before: np.ndarray | None = None
    all_points_after: np.ndarray | None = None
    import time

    with tqdm.tqdm(range(number)) as progress_bar:
        for number in progress_bar:
            change_detection = ChangeDetection(root=root, number=number)
            match_rate = change_detection.run(n_neighbors=n_neighbors)
            progress_bar.set_description(
                f"[{change_detection.label[0]}] result = {match_rate:.2f}"
            )

            result = (
                ChangeType.REMOVE if match_rate < threshold else ChangeType.NO_CHANGE
            )
            y_prediction.append(result)
            # time.sleep(0.5)

            # Load the point cloud for the first time
            if all_points_before is None:
                point_cloud_before_all = o3d.io.read_point_cloud(
                    (root / "Before" / "registered.ply").as_posix()
                )
                all_colors_before = np.array(point_cloud_before_all.colors)
                all_points_before = np.array(point_cloud_before_all.points)
                point_cloud_after_all = o3d.io.read_point_cloud(
                    (root / "After" / "registered.ply").as_posix()
                )
                all_colors_after = np.array(point_cloud_after_all.colors)
                all_colors_after = np.array(point_cloud_after_all.points)

            all_colors_before, all_colors_after = change_detection.color(
                all_colors_before,
                all_colors_after,
                result,
                # y_true[number],
                ChangeType.REMOVE,
            )

        # Save the colored point clouds
        if all_points_before is not None and all_colors_before is not None:
            utils.save_point_cloud(
                all_points_before,
                all_colors_before,
                root / "Before",
                "colored.ply",
            )
        if all_points_after is not None and all_colors_after is not None:
            utils.save_point_cloud(
                all_colors_after,
                all_colors_after,
                root / "After",
                "colored.ply",
            )

    # y_true = y_true[len(y_prediction)]
    y_true = np.ones_like(y_prediction)
    confusion_matrix = metrics.confusion_matrix(y_true, y_prediction)
    print(confusion_matrix)
    classification_report = metrics.classification_report(y_true, y_prediction)
    print(classification_report)

    sns.heatmap(confusion_matrix, square=True, annot=True, cmap="Blues")
    plt.savefig(root / "confusion_matrix.svg")


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Change Detection")
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the processed dataset.",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        help="Number of neighbors.",
        type=int,
        default=Parameter.cd_neighbors,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Threshold of match rate.",
        type=float,
        default=Parameter.cd_threshold,
    )

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path

    if path is None:
        root = pathlib.Path("../dataset/processed/1")
    else:
        root = pathlib.Path(path)

    main(root, args.neighbors, args.threshold)
