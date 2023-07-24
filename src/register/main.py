"""Module for point cloud registration."""

import copy
import pathlib
from argparse import ArgumentParser

import open3d as o3d

from src import utils


class Registration:
    """Registers two point clouds"""

    def __init__(
        self,
        root: pathlib.Path,
        voxel_size: float = 0.05,
    ):
        """Initialize the class.

        Parameters
        ----------
        root : pathlib.Path
            Path to the dataset that contains `Before` and `After` directories.
        voxel_size : float, default 0.05
            Voxel size for downsampling.
        """
        self.before_path = utils.get_point_cloud_path(root / "Before", kind="whole")
        self.after_path = utils.get_point_cloud_path(root / "After", kind="whole")

        self.point_cloud_before = o3d.io.read_point_cloud(self.before_path.as_posix())
        self.point_cloud_after = o3d.io.read_point_cloud(self.after_path.as_posix())
        self.voxel_size: float = voxel_size

        print(" Before ".center(40, "-"))
        self.downsampled_before, self.fpfh_before = self.preprocess(
            self.point_cloud_before
        )
        print(" After ".center(40, "-"))
        self.downsampled_after, self.fpfh_after = self.preprocess(
            self.point_cloud_after
        )

        self.result_ransac = self.fast_ransac()
        self.result_icp = self.icp()

    def preprocess(
        self, point_cloud: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Applies downsampling to the given point cloud, and computes
        1. Normals
        2. FPFH features

        Parameters
        ----------
        point_cloud : o3d.geometry.PointCloud
            Point cloud to be preprocessed

        Returns
        -------
        point_cloud_downsampled : o3d.geometry.PointCloud
            Downsampled point cloud
        point_cloud_fpfh : o3d.pipelines.registration.Feature
            FPFH features of the downsampled point cloud
        """
        # Downsampling
        point_cloud_downsampled = point_cloud.voxel_down_sample(self.voxel_size)

        # Estimation of normals
        radius_normal: float = self.voxel_size * 2
        kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        point_cloud_downsampled.estimate_normals(kdt_n)

        # Compute FPFH features
        radius_feature: float = self.voxel_size * 5
        kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        point_cloud_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud_downsampled, kdt_f
        )

        print(f"{point_cloud} downsampled to {point_cloud_downsampled}")

        return point_cloud_downsampled, point_cloud_fpfh

    def fast_ransac(self) -> o3d.pipelines.registration.RegistrationResult:
        """Global registration by RANSAC.

        Returns
        -------
        result : o3d.pipelines.registration.RegistrationResult
            Result of global registration
        """

        distance_threshold: float = 0.5 * self.voxel_size
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source=self.downsampled_before,
            target=self.downsampled_after,
            source_feature=self.fpfh_before,
            target_feature=self.fpfh_after,
            option=option,
        )

        return result

    def icp(self) -> o3d.pipelines.registration.RegistrationResult:
        """Local refinement by ICP.

        Returns
        -------
        result : o3d.pipelines.registration.RegistrationResult
            Result of local refinement
        """
        estimation_method = (
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        )
        distance_threshold: float = 2 * self.voxel_size

        result = o3d.pipelines.registration.registration_generalized_icp(
            source=self.downsampled_before,
            target=self.downsampled_after,
            max_correspondence_distance=distance_threshold,
            init=self.result_ransac.transformation,
            estimation_method=estimation_method,
        )

        return result

    def visualize(self, mode: str = "global"):
        """Visualize the point clouds before and after registration.

        Parameters
        ----------
        mode : str, default "global"
            Mode of registration. Either "global" or "local".
        """
        window_name: str = (
            "After Global Registration"
            if mode == "global"
            else "After Local Refinement"
        )
        transformation = (
            self.result_ransac.transformation
            if mode == "global"
            else self.result_icp.transformation
        )
        utils.visualize_models(
            self.downsampled_before,
            self.downsampled_after,
            transformation,
            window_name=window_name,
        )

    def save(self):
        """Saves the registered point clouds as PLY files."""

        # Before
        register_before_path = utils.get_point_cloud_path(
            self.before_path.parent, kind="registered"
        )

        print(f"Saving {register_before_path} ...", end="")
        point_cloud_before = copy.deepcopy(self.point_cloud_before)
        point_cloud_before.transform(self.result_icp.transformation)
        o3d.io.write_point_cloud(
            register_before_path.as_posix(),
            point_cloud_before,
            write_ascii=True,
            compressed=True,
            print_progress=True,
        )
        print("Done")

        # After
        register_after_path = utils.get_point_cloud_path(
            self.after_path.parent, kind="registered"
        )

        print(f"Saving {register_after_path} ...", end="")
        o3d.io.write_point_cloud(
            register_after_path.as_posix(),
            self.point_cloud_after,
            write_ascii=True,
            compressed=True,
            print_progress=True,
        )
        print("Done")


if __name__ == "__main__":
    # Setup command line arguments
    parser = ArgumentParser(description="Panoptic Segmentation")
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the dataset that contains `Before` and `After` directories.",
        type=str,
    )
    parser.add_argument("-s", "--size", help="Voxel size", type=float, default=0.1)

    # Parse command line arguments
    args = parser.parse_args()
    path = args.path

    if path is None:
        path = pathlib.Path("../../dataset/processed/1")
    else:
        path = pathlib.Path(path)

    registration = Registration(path, voxel_size=args.size)
    registration.visualize(mode="global")
    registration.visualize(mode="local")
    registration.save()
