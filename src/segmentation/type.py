"""Type definitions for segmentation module."""

from typing import TypedDict

import torch


class InputType(TypedDict):
    """Type definition for input image.

    image: torch.Tensor
        Input image.
    height: int
        Height of the input image.
    width: int
        Width of the input image.
    """

    image: torch.Tensor
    height: int
    width: int


class InfoType(TypedDict):
    """Type definition for instance information.

    count: int
        Number of instances of the same label.
    unpair_list: list[list[str]]
        List of list of instance labels that should be separated.

        For example, if `unpair_list` is `[['a', 'b'], ['c', 'd', 'e']]`, it means:
        - Instances with label `a` and `b` should be separated from each other.
        - Instances with label `c`, `d`, and `e` should be separated from each other.
    """

    count: int
    unpair_list: list[list[str]]


class SegmentInfoType(TypedDict):
    """Type definition for result of segmentation.

    id: int
        ID of the segment.
    category_id: int
        Category ID of the segment.
    isthing: bool
        Whether the segment is thing or not.
    """

    id: int
    category_id: int
    isthing: bool


class ThingSegmentType(SegmentInfoType):
    """Type definition for result of segment with `isthing` being `True`.

    score: float
        Score of the segment label.
    instance_id: int
        ID of the instance.
    """

    score: float
    instance_id: int


class StuffSegmentType(SegmentInfoType):
    """Type definition for result of segment with `isthing` being `False`.

    area: float
        Area of the segment.
    """

    area: int
