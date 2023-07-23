"""Path utilities."""

import pathlib
from typing import Literal


def get_label_path(
    dataset_path: pathlib.Path,
    number: int | None = None,
) -> pathlib.Path:
    """Get the path to the label directory of the dataset. If `number` is specified,
    the path to the `number`-th label file is returned.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.
    number : int, default None
        The label number. If specified, the path to the `number`-th label file is returned.
        Otherwise, the path to the label directory is returned.

    Returns
    -------
    label_path : pathlib.Path
        Path to the label directory of the dataset. If `number` is specified, the path to
        the `number`-th label file is returned.
    """
    if number is not None:
        return dataset_path / "Label" / f"label_{ number}.npy"
    return dataset_path / "Label"


def get_rgb_path(
    dataset_path: pathlib.Path,
    number: int | None = None,
    extension: Literal["jpg", "png"] = "png",
) -> pathlib.Path:
    """Get the path to the rgb directory of the dataset. If `number` is specified,
    the path to the `number`-th rgb file is returned.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.
    number : int, default None
        The rgb number. If specified, the path to the `number`-th rgb file is returned.
        Otherwise, the path to the rgb directory is returned.
    extension : Literal["jpg", "png"], default "png"
        The extension of the rgb file.

    Returns
    -------
    rgb_path : pathlib.Path
        Path to the rgb directory of the dataset. If `n` is specified, the path to
        the `n`-th rgb file is returned.
    """
    if number is not None:
        return dataset_path / "RGB" / f"rgb_{number}.{extension}"
    return dataset_path / "RGB"


def get_depth_path(
    dataset_path: pathlib.Path, number: int | None = None
) -> pathlib.Path:
    """Get the path to the depth directory of the dataset. If `number` is specified,
    the path to the `number`-th depth file is returned.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.
    number : int, default None
        The depth number. If specified, the path to the `number`-th depth file is returned.
        Otherwise, the path to the depth directory is returned.

    Returns
    -------
    depth_path : pathlib.Path
        Path to the depth directory of the dataset. If `number` is specified, the path to
        the `number`-th depth file is returned.
    """
    if number is not None:
        return dataset_path / "Frame" / f"frame_{number}.json"
    return dataset_path / "Frame"


def get_confidence_path(
    dataset_path: pathlib.Path,
    number: int | None = None,
    extension: Literal["jpg", "png"] = "png",
) -> pathlib.Path:
    """Get the path to the confidence directory of the dataset. If `number` is specified,
    the path to the `number`-th confidence file is returned.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset.
    number : int, default None
        The confidence number. If specified, the path to the `number`-th confidence file is
        returned. Otherwise, the path to the confidence directory is returned.
    extension : Literal["jpg", "png"], default "png"
        The extension of the confidence file.

    Returns
    -------
    confidence_path : pathlib.Path
        Path to the confidence directory of the dataset. If `number` is specified, the path to
        the `number`-th confidence file is returned.
    """
    if number is not None:
        return dataset_path / "Confidence" / f"confidence_{number}.{extension}"
    return dataset_path / "Confidence"


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
