"""Configuration file for the project."""


class Parameter:
    """Parameter class for the project."""

    # Panoptic segmentation
    segmentation_step: int = 1

    # Reconstruction
    num_samples_per_frame: int = 5_000
    reconstruction_step: int = 1

    # Merge
    merge_neighbors: int = 30
    merge_distance_threshold: float = 0.02

    # Extract
    used_labels: set[str] = {
        "book",
        "bottle",
        "cup",
        "chair",
        "keyboard",
        "laptop",
        "cell phone",
    }
    min_points: int = 1_000
