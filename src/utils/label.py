"""Utility functions for labels."""


def get_object_label(instance_label: str) -> str:
    """Get the object label from the instance label.

    Parameters
    ----------
    instance_label : str
        Instance label.

    Returns
    -------
    object_label : str
        Object label.
    """
    return instance_label.split("_")[0]
