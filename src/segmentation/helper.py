"""Helper functions for segmentation."""

import itertools

from src.segmentation.type import InfoType


def get_unpair_set(instance_info: dict[str, InfoType]) -> set[tuple[str, str]]:
    unpair_set: set[tuple[str, str]] = set()

    for label in instance_info.values():
        for unpair_list in label["unpair_list"]:
            if len(unpair_list) < 2:
                continue

            unpair_set |= set(itertools.combinations(unpair_list, 2))

    return unpair_set
