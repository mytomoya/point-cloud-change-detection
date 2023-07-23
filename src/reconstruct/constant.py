"""Constants used for point cloud reconstruction."""

from enum import IntEnum

import numpy as np

# When converting image coordinate -> NDC, Y and Z axes should be flipped
# https://github.com/Tomoya-Matsubara/RGB-D-Scan-with-ARKit/blob/main/ScanRGBD/Shaders/UnprojectShader.metal#L13-L19
flipYZ = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)

# https://github.com/Tomoya-Matsubara/RGB-D-Scan-with-ARKit/blob/main/ScanRGBD/PointCloud.swift#L36-L43
angle = np.radians(90)
device_transform_matrix = np.array(
    [
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


class Confidence(IntEnum):
    """Confidence level of the depth map."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
