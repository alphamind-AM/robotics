"""Forward and inverse kinematics for the 4-DOF arm.

`move_c` and `trajectory.follow_line` call these directly. Keep the signatures.

    forward([j1, j2, j3, j4]) -> [x, y, z, pitch_deg]
    inverse(x, y, z, pitch_deg) -> [j1, j2, j3, j4]

Angles in and out are degrees. Use one length unit for every LINK_* and for
x, y, z (millimetres is fine). Joint order is the same as MyRobot.move_j:
j1, j2, j3, j4.

Forward kinematics is the product of the four frames in DH_TABLE. Fill every
line marked TODO. A `None` left in place raises NotImplementedError so an
unfinished equation cannot command the arm.
"""

import math

import numpy as np


# --- Robot geometry -------------------------------------------------------
# Lengths in the same unit as x, y, z.

LINK_1 = 50  # base to shoulder, along z
LINK_2 = 93  # shoulder to elbow
LINK_3 = 93  # elbow to wrist
LINK_4 = 50  # wrist to the commanded point

# Added to each servo angle before the DH table, if servo zero is not
# kinematic zero. Leave a joint at 0.0 when the two zeros already match.
JOINT_ZERO_DEG = np.array([
    0,#-180.0,  # j1
    0,#90.0,   # j2
    0.0,   # j3
    0,#90.0,   # j4
], dtype=float)

# Standard DH, one row per joint: a, alpha_deg, d, theta_offset_deg.
# The angle used for row i is joint_i + JOINT_ZERO_DEG[i] + theta_offset.
# Joint 2's offset puts 0° straight up: z = LINK_1 + LINK_2*cos(q2) + ...
DH_TABLE = np.array([
    [0.0,     90.0, LINK_1, 0.0],
    [LINK_2,   0.0, 0.0,    90.0],
    [LINK_3,   0.0, 0.0,    0.0],
    [LINK_4,   0.0, 0.0,    0.0],
], dtype=float)

# Planar arm has two solutions. Set True for the other elbow branch.
ELBOW_UP = False


def cosd(angle_deg):
    return math.cos(math.radians(angle_deg))


def sind(angle_deg):
    return math.sin(math.radians(angle_deg))


def atand2(y, x):
    """atan2 in degrees."""
    return math.degrees(math.atan2(y, x))


def acosd(value):
    """acos in degrees. `value` must already be clamped to [-1, 1]."""
    return math.degrees(math.acos(value))


def signed_deg(angle):
    """Fold degrees into (-180, 180]. The servos use this signed range, not 0-360."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def _require(names, values):
    missing = [name for name, value in zip(names, values) if value is None]
    if missing:
        raise NotImplementedError(
            "Fill in " + ", ".join(missing) + " in robot/kinematics.py"
        )


def dh_transform(a, alpha_deg, d, theta_deg):
    """One standard DH frame: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ca, sa = cosd(alpha_deg), sind(alpha_deg)
    ct, st = cosd(theta_deg), sind(theta_deg)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


def _arm_frames(joint_angles_deg):
    """Base frame plus the four DH frames, each as a 4x4 pose in the base."""
    joints = np.asarray(joint_angles_deg, dtype=float).reshape(4)
    q = joints + JOINT_ZERO_DEG
    pose = np.eye(4)
    frames = [pose]
    for i in range(4):
        a, alpha_deg, d, theta_offset_deg = DH_TABLE[i]
        pose = pose @ dh_transform(a, alpha_deg, d, q[i] + theta_offset_deg)
        frames.append(pose)
    return frames


def forward_transform(joint_angles_deg):
    """Joint angles in degrees -> 4x4 pose of the end effector in the base frame."""
    return _arm_frames(joint_angles_deg)[-1]


def jacobian(joint_angles_deg, point_ee=None):
    """Geometric Jacobian at a joint configuration q.

    joint_angles_deg is [j1, j2, j3, j4] in degrees. Returns a 6x4 matrix.
    Column i is the twist of the chosen point when joint i moves at 1 rad/s
    and the other joints are held still.

    Rows 0:3 are linear velocity in the base frame, in the same length unit
    as the link lengths, per rad/s. Rows 3:6 are the angular velocity of the
    tool, in rad/s per rad/s.

    point_ee is an optional point fixed in the end-effector frame. Leave it
    out for the tip. The camera passes its mount offset here.
    """
    frames = _arm_frames(joint_angles_deg)
    if point_ee is None:
        point = frames[-1][:3, 3]
    else:
        offset = np.ones(4)
        offset[:3] = np.asarray(point_ee, dtype=float).reshape(3)
        point = (frames[-1] @ offset)[:3]

    jac = np.zeros((6, 4))
    for i in range(4):
        axis = frames[i][:3, 2]
        origin = frames[i][:3, 3]
        jac[:3, i] = np.cross(axis, point - origin)
        jac[3:, i] = axis
    return jac


def jacobian_camera(joint_angles_deg):
    """Geometric Jacobian of the camera origin, at joint configuration q.

    Same layout as jacobian(). The camera is fixed to the tool, so the
    angular rows match the end-effector Jacobian. The linear rows use the
    camera origin instead of the tip.
    """
    from robot.calibration import T_EE_CAMERA

    return jacobian(joint_angles_deg, point_ee=T_EE_CAMERA[:3, 3])


def forward(joint_angles_deg):
    """Joint angles in degrees -> [x, y, z, pitch_deg].

    Position is the translation column of the DH product. Pitch is the
    tool axis measured in the arm plane, up from the positive-reach
    horizontal. Straight up is 90°. Leaning past vertical is past 90°.
    """
    pose = forward_transform(joint_angles_deg)
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
    q1 = float(np.asarray(joint_angles_deg, dtype=float).reshape(4)[0] + JOINT_ZERO_DEG[0])
    tool_axis = pose[:3, 0]
    # Positive reach is the direction x = -cos(q1)*reach, y = -sin(q1)*reach.
    along_reach = -tool_axis[0] * cosd(q1) - tool_axis[1] * sind(q1)
    pitch_deg = atand2(tool_axis[2], along_reach)
    return np.array([x, y, z, pitch_deg], dtype=float)


def inverse(x, y, z, pitch_deg):
    """End-effector pose -> [j1, j2, j3, j4] in degrees."""
    _require(
        ["LINK_1", "LINK_2", "LINK_3", "LINK_4"],
        [LINK_1, LINK_2, LINK_3, LINK_4],
    )

    # Positive reach lies along (-cos j1, -sin j1). The opposite base yaw
    # uses a negative reach. Keep whichever one reproduces this pose.
    r_total = math.hypot(x, y)
    z_wrist = (z - LINK_1) - LINK_4 * sind(pitch_deg)
    tool_horizontal = LINK_4 * cosd(pitch_deg)
    if r_total < 1e-9:
        bases = [(0.0, -tool_horizontal)]
    else:
        bases = [
            (atand2(-y, -x), r_total - tool_horizontal),
            (atand2(y, x), -r_total - tool_horizontal),
        ]

    target = np.array([x, y, z, pitch_deg], dtype=float)
    best = None
    best_error = None
    for j1, r_wrist in bases:
        arm = _arm_plane(r_wrist, z_wrist, pitch_deg)
        if arm is None:
            continue
        j2, j3, j4 = arm
        joints = signed_deg(np.array([j1, j2, j3, j4], dtype=float) - JOINT_ZERO_DEG)
        error = _pose_error(forward(joints), target)
        # Two bases can hit one pose. Keep the first unless the other is
        # actually closer, so a straight line does not flip the base by 180°.
        if best_error is None or error < best_error - 1e-4:
            best = joints
            best_error = error
    if best is None:
        raise ValueError("Target point out of mechanical reach")
    return best


def _arm_plane(r_wrist, z_wrist, pitch_deg):
    """Shoulder, elbow, and wrist for one signed horizontal reach.

    The law of cosines is for an elbow measured from the horizontal.
    Joint 2 in the DH table is measured from straight up, so j2 is
    90° minus that angle, and j3 takes the opposite sign. The tool
    pitch then closes with j2 + j3 + j4 = 90° - pitch.
    """
    cos_t3 = (r_wrist ** 2 + z_wrist ** 2 - LINK_2 ** 2 - LINK_3 ** 2) / (2 * LINK_2 * LINK_3)
    if abs(cos_t3) > 1.0 + 1e-6:
        return None
    cos_t3 = max(-1.0, min(1.0, cos_t3))

    theta3 = acosd(cos_t3)
    if ELBOW_UP:
        theta3 = -theta3
    phi = atand2(z_wrist, r_wrist) - atand2(LINK_3 * sind(theta3), LINK_2 + LINK_3 * cosd(theta3))
    j2 = 90.0 - phi
    j3 = -theta3
    j4 = 90.0 - pitch_deg - j2 - j3
    return j2, j3, j4


def _pose_error(pose, target):
    pose = np.asarray(pose, dtype=float).reshape(4)
    target = np.asarray(target, dtype=float).reshape(4)
    position = float(np.linalg.norm(pose[:3] - target[:3]))
    pitch = (pose[3] - target[3] + 180.0) % 360.0 - 180.0
    return position + abs(pitch)


def roundtrip_error(x, y, z, pitch_deg):
    """Pose error after inverse() then forward(). For your own checks."""
    pose = forward(inverse(x, y, z, pitch_deg))
    return pose - np.array([x, y, z, pitch_deg], dtype=float)


def _fmt_pose(pose):
    pose = np.asarray(pose, dtype=float).reshape(4)
    return (
        f"x={pose[0]:.3f}  y={pose[1]:.3f}  z={pose[2]:.3f}  "
        f"pitch={pose[3]:.3f} deg"
    )


def _fmt_joints(joints):
    joints = np.asarray(joints, dtype=float).reshape(4)
    return "  ".join(f"j{i + 1}={angle:.3f}" for i, angle in enumerate(joints))


def preview(joint_angles_deg=None, pose=None, depth=None):
    """Print joint angles and end-effector position. Does not open the robot.

    joint_angles_deg: four servo angles in degrees. forward() is printed.
    pose: (x, y, z, pitch_deg). inverse() is printed, then forward() of that result.
    depth: camera depth in the same unit as the link lengths. Needs the camera
    offset in robot/calibration.py.
    """
    from robot.calibration import camera_depth_to_base

    print("Kinematics check. The robot is not connected.")

    if joint_angles_deg is None and pose is None:
        joint_angles_deg = [0.0, 0.0, 0.0, 0.0]
        print("No angles given, so this uses joints 0 0 0 0.")

    solved = None
    if pose is not None:
        target = np.asarray(pose, dtype=float).reshape(4)
        print(f"target pose:     {_fmt_pose(target)}")
        try:
            solved = np.asarray(inverse(*target), dtype=float).reshape(4)
        except NotImplementedError as exc:
            print(f"inverse:         {exc}")
        else:
            print(f"joint angles:    {_fmt_joints(solved)} deg")
            print(f"end effector:    {_fmt_pose(forward(solved))}")
            error = forward(solved) - target
            print(
                "roundtrip error: "
                f"dx={error[0]:.3f}  dy={error[1]:.3f}  dz={error[2]:.3f}  "
                f"dpitch={error[3]:.3f} deg"
            )

    arm_joints = None
    if joint_angles_deg is not None:
        arm_joints = np.asarray(joint_angles_deg, dtype=float).reshape(4)
        print(f"joint angles:    {_fmt_joints(arm_joints)} deg")
        print(f"end effector:    {_fmt_pose(forward(arm_joints))}")
    elif solved is not None:
        arm_joints = solved

    if depth is None or arm_joints is None:
        return

    print(f"camera depth:    {float(depth):.3f}")
    try:
        seen = camera_depth_to_base(arm_joints, depth)
    except NotImplementedError as exc:
        print(f"camera point:    {exc}")
        return
    print(
        "camera point:    "
        f"x={seen[0]:.3f}  y={seen[1]:.3f}  z={seen[2]:.3f}"
    )
