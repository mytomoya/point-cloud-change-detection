"""Dataset creation pipeline."""

# pylint: disable=redefined-outer-name

import datetime
import pathlib
import time
from argparse import ArgumentParser
from typing import Literal

from src import (
    Extraction,
    MergeLabel,
    PanopticSegmentation,
    Parameter,
    ReconstructPointCloud,
    Registration,
)


def _create(
    source_path: pathlib.Path,
    destination_path: pathlib.Path,
    segmentation_step: int,
    num_samples_per_frame: int,
    extension: Literal["jpg", "png"],
    reconstruction_step: int,
    merge_neighbors: int,
    merge_distance_threshold: float,
    used_labels: set[str],
    min_points: int,
):
    """Create the dataset by applying the following steps:
    1. Panoptic Segmentation
    2. Reconstruct Point Cloud
    3. Label Merge
    4. Registration
    5. Extract

    See Also
    --------
    `create`
    """
    for period in ("Before", "After"):
        print(f" {period} ".center(30, "-"))

        source_path_ = source_path / period
        destination_path_ = destination_path / period

        # Panoptic Segmentation
        panoptic_segmentation = PanopticSegmentation(dataset_path=source_path_)
        panoptic_segmentation.run(step=segmentation_step)

        # Reconstruct Point Cloud
        reconstruct = ReconstructPointCloud(
            root=source_path_,
            num_samples_per_frame=num_samples_per_frame,
            extension=extension,
            step=reconstruction_step,
        )
        reconstruct.run()

        # Label Merge
        start_time = time.perf_counter()
        merge_label = MergeLabel(
            source_path=source_path_,
            out_path=destination_path_,
            n_neighbors=merge_neighbors,
            distance_threshold=merge_distance_threshold,
        )
        merge_label.merge()
        merge_label.relabel()
        merge_label.save()
        end_time = time.perf_counter()
        elapsed_time = datetime.timedelta(seconds=end_time - start_time)
        print(f"Merging Elapsed: {elapsed_time}")
        print("-" * 30)

    # Registration
    registration = Registration(root=destination_path, voxel_size=0.1)
    registration.save()

    # Extract
    start_time = time.perf_counter()
    extraction = Extraction(
        root=destination_path, used_labels=used_labels, min_points=min_points
    )
    extraction.extract()
    end_time = time.perf_counter()
    elapsed_time = datetime.timedelta(seconds=end_time - start_time)
    print(f"Extraction Elapsed: {elapsed_time}")


def create(
    source_path: pathlib.Path,
    destination_path: pathlib.Path,
    segmentation_step: int,
    num_samples_per_frame: int,
    extension: Literal["jpg", "png"],
    reconstruction_step: int,
    merge_neighbors: int,
    merge_distance_threshold: float,
    used_labels: set[str],
    min_points: int,
):
    """Create the dataset.

    Parameters
    ----------
    source_path : pathlib.Path
        Path to the source dataset. It should contain either "Before"/"After" or
        "1"/"2"/... folders.
    destination_path : pathlib.Path
        Path to the destination dataset. Like the source dataset, it should contain either
        "Before"/"After" or "1"/"2"/... folders.
    segmentation_step : int
        Segmentation step.
    num_samples_per_frame : int
        Number of samples per frame.
    extension : Literal["jpg", "png"]
        Image extension.
    reconstruction_step : int
        Reconstruction step.
    merge_neighbors : int
        Number of neighboring points to consider when merging labels.
    merge_distance_threshold : float
        Threshold of distance to consider when merging labels.
    used_labels : set[str]
        Set of labels that are to be used. Labels not in this set are ignored.
    min_points : int
        Minimum number of points in a point cloud. If point cloud extracted from extracted has
        less points than this value, it is discarded.

    See Also
    --------
    `_create`
    """
    if (source_path / "1").exists():
        for path in source_path.iterdir():
            _create(
                path,
                destination_path / path.name,
                segmentation_step,
                num_samples_per_frame,
                extension,
                reconstruction_step,
                merge_neighbors,
                merge_distance_threshold,
                used_labels,
                min_points,
            )
    else:
        _create(
            source_path,
            destination_path,
            segmentation_step,
            num_samples_per_frame,
            extension,
            reconstruction_step,
            merge_neighbors,
            merge_distance_threshold,
            used_labels,
            min_points,
        )


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Panoptic Segmentation")
    parser.add_argument("--source", help="Path to the source dataset", type=str)
    parser.add_argument("--dest", help="Path to the destination dataset", type=str)
    parser.add_argument(
        "-ss",
        "--segmentation_step",
        help="Segmentation step",
        type=int,
        default=Parameter.segmentation_step,
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        help="Number of samples per frame",
        type=int,
        default=Parameter.num_samples_per_frame,
    )
    parser.add_argument(
        "-e",
        "--extension",
        help="Image extension",
        type=str,
        default="jpg",
    )
    parser.add_argument(
        "-rs",
        "--reconstruction_step",
        help="Reconstruction step",
        type=int,
        default=Parameter.reconstruction_step,
    )
    parser.add_argument(
        "-mn",
        "--merge_neighbors",
        help="Number of neighbors used to merge",
        type=int,
        default=Parameter.merge_neighbors,
    )
    parser.add_argument(
        "-t",
        "--merge_distance_threshold",
        help="Distance threshold used to merge",
        type=float,
        default=Parameter.merge_distance_threshold,
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
    source = args.source
    destination = args.dest

    if source is None:
        source_path = pathlib.Path("../dataset/raw")
    else:
        source_path = pathlib.Path(source)

    if destination is None:
        destination = pathlib.Path("../dataset/processed")
    else:
        destination = pathlib.Path(destination)

    labels = set(args.labels) if args.labels is not None else Parameter.used_labels

    create(
        source_path,
        destination,
        args.segmentation_step,
        args.num_samples,
        args.extension,
        args.reconstruction_step,
        args.merge_neighbors,
        args.merge_distance_threshold,
        labels,
        args.min,
    )
