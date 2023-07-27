"""Type definitions."""

from enum import IntEnum
from typing import Literal, TypedDict

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


class AnnotationItemType(TypedDict):
    """Type definition for annotation item.

    id: int
        ID of the item.
    object: str
        Object name.
    change: Literal["removed", "no change", "]
        Change type.
    """

    id: int
    object: str
    change: Literal["removed", "no change", ""]


class AnnotationType(TypedDict):
    """Type definition for annotation.

    point_cloud: list[AnnotationItemType]
        List of annotation items.
    """

    point_cloud: list[AnnotationItemType]


class ChangeType(IntEnum):
    """Type definition for change type. There are two types of change: no
    change and remove.
    """

    NO_CHANGE = 0
    REMOVE = 1
