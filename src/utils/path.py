"""Path utilities."""

import pathlib


def get_label_path(dataset_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the label directory of the dataset.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.

    Returns
    -------
    label_path : pathlib.Path
        Path to the label directory of the dataset.
    """
    return dataset_path / "Label"


def get_rgb_path(dataset_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the rgb directory of the dataset.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.

    Returns
    -------
    rgb_path : pathlib.Path
        Path to the rgb directory of the dataset.
    """
    return dataset_path / "RGB"


def get_unpair_path(dataset_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the `unpair.json` of the dataset.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.

    Returns
    -------
    unpair_path : pathlib.Path
        Path to the `unpair.json` of the dataset.
    """
    return dataset_path / "unpair.json"
