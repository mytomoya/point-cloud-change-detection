"""Configuration file for the project."""


class Parameter:
    """Parameter class for the project."""

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
