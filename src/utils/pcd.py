"""Utility functions for point cloud processing."""

import copy
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

    save_path = root / save_file

    print(f"Saving {save_file}... ", end="")
    o3d.io.write_point_cloud(
        save_path.as_posix(),
        pcd,
        write_ascii=True,
        compressed=True,
        print_progress=True,
    )
    print("Saved!")


def visualize_models(
    point_cloud1: o3d.geometry.PointCloud,
    point_cloud2: o3d.geometry.PointCloud,
    transformation: np.ndarray = np.identity(4),
    window_name: str = "Point Cloud",
):
    """Visualize two point clouds.

    Parameters
    ----------
    point_cloud1 : o3d.geometry.PointCloud
        First point cloud.
    point_cloud2 : o3d.geometry.PointCloud
        Second point cloud.
    transformation : np.ndarray, default np.identity(4)
        Transformation matrix.
    window_name : str, default "Point Cloud"
        Name of the window.
    """
    point_cloud1_copy = copy.deepcopy(point_cloud1)
    point_cloud2_copy = copy.deepcopy(point_cloud2)

    point_cloud1_copy.transform(transformation)

    o3d.visualization.draw_geometries(  # pylint: disable=no-member
        [point_cloud1_copy, point_cloud2_copy], window_name=window_name
    )
