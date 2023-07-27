import json
import pathlib

from src.type import AnnotationItemType, AnnotationType, ChangeType


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


def get_ground_truth(dataset_path: pathlib.Path) -> list[ChangeType]:
    """Get the ground truth from the annotation file.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the dataset. The annotation file is located in the dataset.

    Returns
    -------
    y_true : list[ChangeType]
        Ground truth. 1 if the point cloud is removed, 0 otherwise.
    """
    annotation_path = dataset_path / "annotation.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"{annotation_path.name} does not exist.")

    y_true: list[ChangeType] = []
    json_data = load(annotation_path, create=False)
    data = json_data["point_cloud"]

    n_items: int = len(data)
    for i in range(n_items):
        y_true.append(
            ChangeType.REMOVE if data[i]["change"] == "remove" else ChangeType.NO_CHANGE
        )

    return y_true
