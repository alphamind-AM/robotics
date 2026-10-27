"""Camera mount relative to the end effector.

The camera is fixed to the tip. T_EE_CAMERA is that rigid transform, written
in the end-effector frame from forward_transform(), not the base frame.
camera_transform() is T_base_ee @ T_EE_CAMERA.
"""

import numpy as np


# Optical center of the camera, in the end-effector frame.
# Columns of forward_transform() are that frame: +X is the way the gripper
# points, and +Y, +Z complete the right-handed tool frame. Millimetres,
# the same unit as LINK_*.
#
# The mount is a pure translation: -15 mm along tool X, +45 mm along tool Y.
T_EE_CAMERA = np.array([
    [1.0, 0.0, 0.0, -15.0],
    [0.0, 1.0, 0.0,  45.0],
    [0.0, 0.0, 1.0,   0.0],
    [0.0, 0.0, 0.0,   1.0],
], dtype=float)

# Direction the camera looks. The mount has no rotation, so this vector is the
# same in the camera frame and the end-effector frame. Straight out of the
# gripper is +X. Change this if the camera points elsewhere.
CAMERA_VIEW_AXIS = np.array([1.0, 0.0, 0.0], dtype=float)


def camera_transform(joint_angles_deg):
    """4x4 pose of the camera in the base frame.

    The translation in T_EE_CAMERA stays in the tool frame, so it moves
    with the end effector.
    """
    from robot.kinematics import forward_transform

    return forward_transform(joint_angles_deg) @ T_EE_CAMERA


def camera_depth_to_base(joint_angles_deg, depth):
    """Depth along the camera view -> [x, y, z] in the robot base frame.

    joint_angles_deg is the arm pose the picture was taken at.
    depth is the distance the camera measured, in millimetres.
    The seen point is T_base_ee @ T_EE_CAMERA @ [depth along the view, 1].
    """
    from robot.kinematics import forward_transform

    view = np.asarray(CAMERA_VIEW_AXIS, dtype=float).reshape(3)
    length = float(np.linalg.norm(view))
    if length == 0.0:
        raise ValueError("CAMERA_VIEW_AXIS must be a non-zero direction")
    view = view / length

    point_cam = np.ones(4)
    point_cam[:3] = float(depth) * view
    point_base = forward_transform(joint_angles_deg) @ T_EE_CAMERA @ point_cam
    return point_base[:3]


def pixel_to_point(pixel, image_shape):
    """Map a camera pixel to a point the arm can aim at.

    pixel is (u, v). image_shape is the frame's (height, width) or
    (height, width, channels). Return (x, y, z, pitch_deg) in the robot frame.

    This needs the camera mount and its pixel-to-direction scale. Until that
    calibration is filled in, pointing at a face has nowhere to aim.
    """
    raise NotImplementedError(
        "Add the camera mounting geometry in robot/calibration.py. "
        "pixel_to_point should return (x, y, z, pitch_deg)."
    )
