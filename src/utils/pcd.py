"""Utility functions for point cloud processing."""

import pathlib

import numpy as np
import open3d as o3d


def save_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    root: pathlib.Path,
    save_file: str = "point_cloud.ply",
):
    """Save the point cloud as a PLY file.

    Parameters
    ----------
    points : np.ndarray
        Point cloud.
    colors : np.ndarray
        Color of each point.
    root : pathlib.Path
        Path to the dataset.
    save_file : str, default "point_cloud.ply"
        Name of the ply file to save.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ply_directory = root / "PLY"

    # If the `PLY` directory does not exist, create it
    ply_directory.mkdir(exist_ok=True)

    save_path = ply_directory / save_file

    print(f"Saving {save_file}... ", end="")
    o3d.io.write_point_cloud(
        save_path.as_posix(),
        pcd,
        write_ascii=True,
        compressed=True,
        print_progress=True,
    )
    print("Saved!")
