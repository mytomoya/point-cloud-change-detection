"""Module for merging instance labels of the same instance into one."""

import datetime
import json
import pathlib
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import open3d as o3d
import tqdm
from sklearn import neighbors

from src import Parameter, utils
from src.merge.union_find import UnionFind


class MergeLabel:
    """Merges instance labels of the same instance into one"""

    def __init__(
        self,
        source_path: pathlib.Path,
        out_path: pathlib.Path,
        n_neighbors: int = Parameter.merge_neighbors,
        distance_threshold: float = Parameter.merge_distance_threshold,
    ):
        """Initialization.

        Parameters
        ----------
        source_path : `pathlib.Path`
            Path to the source dataset.
        out_path : `pathlib.Path`
            Path to the destination dataset.
        n_neighbors : `int`, default Parameter.merge_neighbors
            Number of neighboring points to consider when merging labels.
        distance_threshold : `float`, default Parameter.merge_distance_threshold
            Threshold of distance to consider when merging labels.
        """

        self.source_path = source_path
        self.out_path = out_path
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold

        # Point Cloud
        point_cloud_path = utils.get_point_cloud_path(source_path)
        self.point_cloud = o3d.io.read_point_cloud(point_cloud_path.as_posix())
        self.points = np.array(self.point_cloud.points)

        # Labels
        label_path = utils.get_label_out_path(source_path)
        self.labels = np.load(label_path.as_posix(), allow_pickle=True)
        self.unique_labels = np.unique(self.labels)
        self.index2label: dict[int, str] = dict(enumerate(self.unique_labels))
        self.label2index: dict[str, int] = {
            label: index for index, label in enumerate(self.unique_labels)
        }

        # Convert label to index
        def convert_to_index(label: str) -> int:
            return self.label2index[label]

        np_convert_to_index = np.frompyfunc(convert_to_index, 1, 1)
        self.label_indices = np_convert_to_index(self.labels).astype(np.int32)
        self.unique_label_indices = np.unique(self.label_indices)
        self.new_labels = self.labels.copy()

        # Unpair Set
        json_path = utils.get_unpair_path(source_path)
        with open(json_path, mode="r", encoding="utf-8") as f:
            unpair_list: list[list[str]] = json.load(f)
        self.unpair_set: set[tuple[str, str]] = set(map(tuple, unpair_list))
        self.unpair_dictionary: dict[int, set[int]] = defaultdict(set)

        self.initialize_unpair_dictionary()

        self.union_find = UnionFind(
            n=len(self.points), label_indices=self.label_indices
        )
        self.initialize_union_find()

        self.knn = neighbors.NearestNeighbors(n_jobs=-1)
        self.knn.fit(self.points)

    def initialize_unpair_dictionary(self):
        """Initializes the unpair dictionary."""

        for label1, label2 in tqdm.tqdm(
            self.unpair_set, desc="[Making Unpair Dictionary]", leave=False
        ):
            index1: int = self.label2index.get(label1, -1)
            index2: int = self.label2index.get(label2, -1)
            if -1 in (index1, index2):
                continue

            self.unpair_dictionary[index1].add(index2)
            self.unpair_dictionary[index2].add(index1)

    def initialize_union_find(self):
        """Combines points with the same label together in Union-Find."""

        for label_index in tqdm.tqdm(
            self.unique_label_indices, desc="[Initialize Union-Find]"
        ):
            label: str = self.index2label[label_index]
            if len(label) == 0:
                continue

            indices: npt.NDArray[np.int32] = np.where(
                self.label_indices == label_index
            )[0]

            index1: int = indices[0]

            for index2 in tqdm.tqdm(indices[1:], desc=f"({label})", leave=False):
                self.union_find.union(index1, index2)

    def merge(self):
        """Merges labels with the same instance label whose points are within a threshold
        distance.
        """

        # Compute neighbors
        start: float = time.perf_counter()
        print("Compute neighbors... ", end="")
        distances, indices = self.knn.kneighbors(
            self.points, n_neighbors=self.n_neighbors
        )
        end: float = time.perf_counter()
        elapsed = datetime.timedelta(seconds=end - start)
        print(f"elapsed: {elapsed}")

        # Filter by threshold
        print(
            f"Filtering by threshold {self.distance_threshold}: {distances.size:,} -> ",
            end="",
        )
        point_indices1, nearest_indices = np.where(distances < self.distance_threshold)
        point_indices2 = indices[point_indices1, nearest_indices]
        print(f"{len(point_indices1):,}")

        # Get label indices
        print("Getting label indices -> ", end="")
        label_indices1 = self.label_indices[point_indices1]
        label_indices2 = self.label_indices[point_indices2]
        print(f"{len(label_indices1):,} labels")

        # Remove the same label indices pairs
        print(
            f"Removing pairs of the same label: {len(label_indices1):,} -> ",
            end="",
        )
        condition = np.where(label_indices1 != label_indices2)[0]
        point_indices1 = point_indices1[condition]
        point_indices2 = point_indices2[condition]
        print(f"{len(point_indices1):,} labels")

        # Remove duplicate pairs
        print(f"Removing duplicate pairs: {len(point_indices1):,} -> ", end="")
        indices_pair = np.stack([point_indices1, point_indices2])
        indices_pair, condition = np.unique(indices_pair, axis=1, return_index=True)
        point_indices1 = point_indices1[condition]
        point_indices2 = point_indices2[condition]
        print(f"{indices_pair.shape[1]:,} labels")

        # Get object labels
        print("Getting object labels")

        def convert_to_label(index: int) -> str:
            return self.index2label[self.label_indices[index]].split("_")[0]

        np_convert_to_label = np.frompyfunc(convert_to_label, 1, 1)
        object_label1, object_label2 = np_convert_to_label(indices_pair)

        # Remove the same instance label pairs
        print(
            f"Filtering pairs of the same object label: {len(point_indices1):,}-> ",
            end="",
        )
        condition = np.where(object_label1 == object_label2)[0]
        point_indices1 = point_indices1[condition]
        point_indices2 = point_indices2[condition]
        print(f"{len(point_indices1):,} labels")

        for index1, index2 in tqdm.tqdm(
            zip(point_indices1, point_indices2),
            total=point_indices1.shape[0],
            desc="[Union Labels]",
        ):
            if (
                len(self.unpair_dictionary[index1] & self.union_find.members(index2))
                > 0
            ):
                continue
            if (
                len(self.unpair_dictionary[index2] & self.union_find.members(index1))
                > 0
            ):
                continue

            self.union_find.union(index1, index2)

    def relabel(self):
        """Renames the labels based on the result of Union-Find."""

        print("Relabeling...", end=" ")
        start: float = time.perf_counter()

        labels = self.labels.copy()
        parents: npt.NDArray[np.int32] = self.union_find.parents.copy()

        labels[parents > 0] = labels[parents[parents > 0]]
        self.new_labels = labels.copy()

        end: float = time.perf_counter()
        elapsed = datetime.timedelta(seconds=end - start)
        print(f"elapsed: {elapsed}")

    def save(self):
        """Saves the renamed labels and the point cloud. The labels are saved in the source
        dataset and the destination dataset as `merged_label.npy`. The point cloud is saved in
        the destination dataset as `whole_point_cloud.ply`.
        """

        save_path = self.source_path / "merged_label"
        np.save(save_path.as_posix(), self.new_labels)

        # Save new labels in the destination dataset
        save_path = self.out_path / "merged_label"
        np.save(save_path.as_posix(), self.new_labels)

        # Save point cloud in the destination dataset
        save_path = self.out_path / "whole_point_cloud.ply"
        o3d.io.write_point_cloud(
            save_path.as_posix(),
            self.point_cloud,
            write_ascii=True,
            compressed=True,
            print_progress=True,
        )


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Instance Segmentation")
    parser.add_argument("-p", "--path", help="Path to the source dataset", type=str)
    parser.add_argument("-o", "--out", help="Path to the destination dataset", type=str)
    parser.add_argument(
        "-n", "--neighbors", help="Number of neighbors", type=int, default=17
    )
    parser.add_argument(
        "-d", "--distance", help="Distance threshold", type=float, default=0.02
    )

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path
    out_path = args.out

    if path is None:
        source_path = pathlib.Path("../../dataset/raw/1/Before")
    else:
        source_path = pathlib.Path(path)

    if out_path is None:
        out_path = pathlib.Path()
    else:
        out_path = pathlib.Path(out_path)

    merge_label = MergeLabel(
        source_path=source_path,
        out_path=out_path,
        n_neighbors=args.neighbors,
        distance_threshold=args.distance,
    )
    merge_label.merge()
    merge_label.relabel()
    merge_label.save()
