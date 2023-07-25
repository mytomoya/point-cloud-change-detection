import json
import pathlib

import numpy as np
import open3d as o3d

from src.type import AnnotationItemType, AnnotationType

# from ._type import ItemType, JsonDataType, LabelsType

# ==============================================================================
# Annotation file manipulation
# ==============================================================================


def load(path: pathlib.Path, create: bool = True) -> AnnotationType:
    """Loads the annotation json file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the json file.
    create : bool, default True
        If True, creates a new json file even if the file exsits.

    Returns
    -------
    json_data : AnnotationType
        Loaded json file.
    """

    # If the json file exsits,
    if not create and path.exists():
        # then loads it
        with open(path, mode="r", encoding="utf-8") as file:
            json_data: AnnotationType = json.load(file)
    # If the json file does not exsit
    else:
        # then create a new one
        json_data = AnnotationType(point_cloud=[])

    return json_data


def add(json_data: AnnotationType, item: AnnotationItemType) -> AnnotationType:
    """Adds the given item to the json_data.

    Parameters
    ----------
    json_data : AnnotationType
        JSON data to which the item is added.
    item : AnnotationItemType
        Item that is added to json data.

    Returns
    -------
    json_data : AnnotationType
        Updated json data.
    """

    if item not in json_data["point_cloud"]:
        json_data["point_cloud"].append(item)

    return json_data


def save(path: pathlib.Path, json_data: AnnotationType):
    """Saves the json data to the given path.

    Parameters
    ----------
    path : Path
        Path to the json file.
    json_data : AnnotationType
        JSON data to be saved.
    """

    with open(path, mode="w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)
