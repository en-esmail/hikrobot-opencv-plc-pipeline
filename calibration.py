from typing import Tuple

import numpy as np

from constants import CalibrationConstants


# ---------------------------------------------------------------------------
# 3. Camera-pixel -> robot-frame calibration
# ---------------------------------------------------------------------------
PIXEL_TO_ROBOT_MATRIX = np.array(CalibrationConstants.PIXEL_TO_ROBOT_MATRIX)


def pixel_to_robot(px: float, py: float) -> Tuple[float, float]:
    """Convert pixel coordinates to robot frame coordinates.

    Uses affine transformation matrix fitted from calibration correspondences.
    See CalibrationConstants.PIXEL_TO_ROBOT_MATRIX for the transformation.

    Args:
        px: Pixel x-coordinate
        py: Pixel y-coordinate

    Returns:
        Tuple of (robot_x, robot_y) in millimeters
    """
    vec = np.array([px, py, 1.0])
    robot_x, robot_y = PIXEL_TO_ROBOT_MATRIX @ vec
    return float(robot_x), float(robot_y)
