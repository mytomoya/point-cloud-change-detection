"""Type definitions."""

from typing import TypedDict

import numpy as np


class DepthMapType(TypedDict):
    """Type definition for depth map.

    width: int
        Width of the depth map.
    height: int
        Height of the depth map.
    values: list[float]
        Values of the depth map flattened to 1D array.
    """

    width: int
    height: int
    values: list[float]


class JsonType(TypedDict):
    """Type definition for JSON frame file.

    depth_map: DepthMapType
        Depth map.
    intrinsic: np.ndarray
        Intrinsic matrix of shape 3 x 3.
    view_matrix: np.ndarray
        View matrix of shape 4 x 4.
    frame_number: int
        Frame number.
    """

    depth_map: DepthMapType
    intrinsic: np.ndarray
    view_matrix: np.ndarray
    frame_number: int
