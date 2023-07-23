"""Utility functions for loading images."""

# pylint: disable=no-member
import json
import pathlib
from typing import Literal

import cv2
import numpy as np

from src.type import DepthMapType, JsonType

from .path import get_confidence_path, get_depth_path, get_rgb_path


def get_depth_map(path: str | pathlib.Path) -> np.ndarray:
    """Loads and returns the depth map.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the depth map file.

    Returns
    -------
    depth_map : np.ndarray
        Depth map.
    """
    with open(path, encoding="utf-8", mode="r") as file:
        data: JsonType = json.load(file)

        depth_data: DepthMapType = data["depth_map"]

        width: int = depth_data["width"]
        height: int = depth_data["height"]
        values: list[float] = depth_data["values"]

        # Reshape the depth map
        depth_map = np.array(values, dtype=np.float32)
        depth_map = np.reshape(depth_map, (height, width, -1))

        # Rotate the depth map
        depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)

    return depth_map


def load_images(
    number: int,
    root: pathlib.Path,
    extension: Literal["jpg", "png"] = "png",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Loads and returns the RGB image, depth map, and confidence map
    of the specified number.

    Parameters
    ----------
    number : int
        Frame number.
    root : pathlib.Path
        Path to the dataset.
    extension : Literal["jpg", "png"], default "png"
        The extension of the rgb/confidence file.

    Returns
    -------
    [image_rgb, depth_map, confidence_map] : Optional[tuple[ImageType, ImageType, ImageType]]
    """

    # RGB
    rgb_path = get_rgb_path(root, number, extension)
    # If the RGB image does not exist, return None
    if not rgb_path.exists():
        return None

    image_bgr = cv2.imread(rgb_path.as_posix()).astype(np.float32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Depth
    depth_path = get_depth_path(root, number)
    # If the depth map does not exist, return None
    if not depth_path.exists():
        return None
    depth_map = get_depth_map(depth_path)
    # Resize
    depth_map = cv2.resize(depth_map, (width, height))

    # Confidence
    confidence_path = get_confidence_path(root, number, extension)
    if not confidence_path.exists():
        return None
    confidence_map = cv2.imread(
        confidence_path.as_posix(),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float32)
    # Resize
    confidence_map = cv2.resize(confidence_map, (width, height))

    # Rotate images
    image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    confidence_map = cv2.rotate(confidence_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_rgb, depth_map, confidence_map


def get_parameters(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads and returns the intrinsic and view matrix.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the json file.

    Returns
    -------
    intrinsic : np.ndarray
        Intrinsic matrix of shape (3, 3).
    view_matrix : np.ndarray
        View matrix of shape (4, 4).
    """
    with open(path, encoding="utf-8", mode="r") as file:
        data: JsonType = json.load(file)

        intrinsic = np.array(data["intrinsic"])
        view_matrix = np.array(data["view_matrix"])

        # squeeze the matrices
        intrinsic = np.squeeze(intrinsic)
        view_matrix = np.squeeze(view_matrix)

        if intrinsic.shape != (3, 3):
            raise ValueError("The shape of the intrinsic matrix is not (3, 3).")
        if view_matrix.shape != (4, 4):
            raise ValueError("The shape of the view matrix is not (4, 4).")

        # column major -> row major
        intrinsic = intrinsic.T
        view_matrix = view_matrix.T

    return intrinsic, view_matrix
