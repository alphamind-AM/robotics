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
CAMERA_VIEW_AXIS = np.array([-1.0, 0.0, 0.0], dtype=float)

# Pixels. None until a calibration fills it. point_ring does not read this:
# it learns pixels per millimetre from a camera shift. pixel_to_table still
# needs a focal length.
FOCAL_PX = None

# Height of the plane the ring lies on, in the base frame. The model origin
# is on the table, so an unraised base uses 0. Measure this if the table
# the ring sits on is not that plane.
RING_PLANE_Z = 0.0

# Columns are image-right, image-down, and the view direction, written in
# the tool frame. The view is tool +X. With the tool pointing straight down,
# image-down is tool -Y, which is base +X (forward). Image-right is -tool Z,
# which is base -Y. The opposite image-down sent the arm backward, away
# from a ring that sat lower in the picture.
CAMERA_AXES_IN_TOOL = np.array([
    [0.0,  0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=float)

# Where the stylus tip appears, in pixels, relative to the image center.
# Positive du is right, positive dv is down. (0, 0) puts the tip on the
# center pixel. A real tip is a 3D point, so one offset is only correct at
# one depth. Replace stylus_pixel() with a projection when the camera scale
# exists; callers pass the offset through and do not need to change.
STYLUS_PIXEL_OFFSET = (0.0, 0.0)


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


def stylus_pixel(image_shape, offset=None):
    """Pixel where the stylus tip is assumed to be.

    image_shape is (height, width) or (height, width, channels).
    offset is (du, dv) from the image center. None uses STYLUS_PIXEL_OFFSET.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    du, dv = STYLUS_PIXEL_OFFSET if offset is None else offset
    return (0.5 * (width - 1) + float(du), 0.5 * (height - 1) + float(dv))


def pixel_travel(target_uv, image_shape, offset=None):
    """Image error from the stylus pixel to a target pixel.

    Returns (du, dv). Positive du means the target is to the right of the
    stylus. Positive dv means it is below the stylus, in image coordinates.

    Driving this to (0, 0) lays the stylus pixel on the target. With the
    camera fixed to the tool, that is the visual error to close. It is a
    pixel count, not a joint move and not a distance in millimetres.
    """
    su, sv = stylus_pixel(image_shape, offset)
    tu, tv = target_uv
    return (float(tu) - su, float(tv) - sv)


def _camera_pose(joint_angles_deg):
    from robot.kinematics import forward_transform

    pose = forward_transform(joint_angles_deg)
    origin = (pose @ T_EE_CAMERA)[:3, 3]
    rotation = pose[:3, :3]
    return origin, rotation


def pixel_to_table(pixel, image_shape, joint_angles_deg, focal_px=None, plane_z=None):
    """Where the camera ray through pixel meets the ring plane.

    pixel is (u, v). joint_angles_deg is the arm pose the picture was taken
    at. Returns (x, y, z) millimetres in the base, or None when the ray does
    not meet the plane in front of the camera.

    focal_px None uses FOCAL_PX. The picture has no distance in it. The plane
    supplies that distance.
    """
    focal = FOCAL_PX if focal_px is None else focal_px
    if focal is None or float(focal) <= 0.0:
        raise ValueError("Set FOCAL_PX in robot/calibration.py before aiming from a pixel.")
    height = RING_PLANE_Z if plane_z is None else float(plane_z)
    u, v = float(pixel[0]), float(pixel[1])
    cx = 0.5 * (int(image_shape[1]) - 1)
    cy = 0.5 * (int(image_shape[0]) - 1)
    optical = np.array([(u - cx) / float(focal), (v - cy) / float(focal), 1.0])
    origin, rotation = _camera_pose(joint_angles_deg)
    direction = rotation @ (CAMERA_AXES_IN_TOOL @ optical)
    if abs(float(direction[2])) < 1e-6:
        return None
    distance = (height - float(origin[2])) / float(direction[2])
    if distance <= 1.0:
        return None
    return origin + distance * direction


def focal_from_point(pixel, image_shape, joint_angles_deg, point_xyz, plane_z=None):
    """Focal length that puts pixel on a ring whose base position is known.

    Returns (f_from_u, f_from_v). Either is None when that axis is too close
    to the image center to measure. A negative value means image-right or
    image-down is flipped in CAMERA_AXES_IN_TOOL.
    """
    height = RING_PLANE_Z if plane_z is None else float(plane_z)
    point = np.array([float(point_xyz[0]), float(point_xyz[1]), height], dtype=float)
    origin, rotation = _camera_pose(joint_angles_deg)
    optical = CAMERA_AXES_IN_TOOL.T @ (rotation.T @ (point - origin))
    if float(optical[2]) <= 1.0:
        return None, None
    u, v = float(pixel[0]), float(pixel[1])
    cx = 0.5 * (int(image_shape[1]) - 1)
    cy = 0.5 * (int(image_shape[0]) - 1)

    def axis(offset, component):
        if abs(offset) < 25.0 or abs(component) < 1e-6:
            return None
        return offset * float(optical[2]) / component

    return axis(u - cx, float(optical[0])), axis(v - cy, float(optical[1]))


def _tool_axes(yaw_deg, pitch_deg):
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    c1, s1 = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    tool_x = np.array([c1 * cp, s1 * cp, sp])
    tool_y = np.array([c1 * sp, s1 * sp, -cp])
    tool_z = np.array([-s1, c1, 0.0])
    return np.column_stack((tool_x, tool_y, tool_z))


def aim_camera_at(point_xyz, pitch_deg=-30.0, standoff_mm=200.0):
    """Tip pose whose camera axis passes through point_xyz.

    The camera looks along tool +X, from standoff_mm away, at pitch_deg.
    Returns (x, y, z, pitch_deg), or None when that pose is unreachable,
    outside the joint limits, or would put a link below 40 mm.
    """
    from robot.arm import JOINT_LIMITS_DEG
    from robot.kinematics import _lowest_link_z, forward_transform, inverse
    from robot import kinematics

    point = np.asarray(point_xyz, dtype=float).reshape(3)
    yaw = float(np.rad2deg(np.arctan2(point[1], point[0])))
    if abs(yaw) > 125.0:
        return None
    rotation = _tool_axes(yaw, pitch_deg)
    # Camera origin = tip + R @ t. Looking along tool +X: origin + s * tool_x = point.
    offset = T_EE_CAMERA[:3, 3]
    tip = point - float(standoff_mm) * rotation[:, 0] - rotation @ offset
    if float(tip[2]) < 40.0:
        return None
    pose = (float(tip[0]), float(tip[1]), float(tip[2]), float(pitch_deg))
    previous = bool(kinematics.ELBOW_UP)
    try:
        best = None
        # The mission repeats this solve with the elbow-up branch.
        for elbow_up in (True,):
            kinematics.ELBOW_UP = elbow_up
            try:
                joints = np.asarray(inverse(*pose), dtype=float).reshape(4)
            except (ValueError, NotImplementedError):
                continue
            limits = JOINT_LIMITS_DEG
            if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
                continue
            if _lowest_link_z(joints) < 40.0:
                continue
            transform = forward_transform(joints)
            origin = (transform @ T_EE_CAMERA)[:3, 3]
            view = transform[:3, 0]
            gap = point - origin
            along = float(np.dot(gap, view))
            if along < 50.0:
                continue
            miss = float(np.linalg.norm(gap - along * view))
            if best is None or miss < best[0]:
                best = (miss, pose)
        if best is None or best[0] > 15.0:
            return None
        return best[1]
    finally:
        kinematics.ELBOW_UP = previous


def shift_pose(joint_angles_deg, distance_mm=18.0, pitch_deg=None, height_mm=None):
    """Slide the camera across the image without changing its distance.

    The slide stays in the arm's vertical plane, so the base does not yaw
    and the tool frame does not spin. height_mm, when given, keeps the tip
    at that height. Returns (x, y, z, pitch_deg), or None when that slide
    does not fit inside the joint limits.
    """
    from robot.arm import JOINT_LIMITS_DEG
    from robot.kinematics import forward, inverse
    from robot import kinematics

    pose = forward(joint_angles_deg)
    _origin, rotation = _camera_pose(joint_angles_deg)
    view = (rotation @ CAMERA_AXES_IN_TOOL)[:, 2]
    if abs(float(view[2])) < 0.25:
        return None
    radial = np.array([float(pose[0]), float(pose[1]), 0.0], dtype=float)
    if float(np.linalg.norm(radial)) < 20.0:
        yaw = np.deg2rad(float(np.asarray(joint_angles_deg, dtype=float).reshape(4)[0]))
        radial = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
    else:
        radial = radial / np.linalg.norm(radial)
    # radial * view_z - up * (radial · view) is perpendicular to the view
    # and has no sideways component, so the optical distance stays put.
    raw = radial * float(view[2]) - np.array([0.0, 0.0, float(np.dot(radial, view))])
    length = float(np.linalg.norm(raw))
    if length < 1e-6:
        return None
    raw = raw / length
    if raw[2] < 0.0:
        raw = -raw
    tip = pose[:3] + float(distance_mm) * raw
    if height_mm is not None:
        tip = np.array([float(tip[0]), float(tip[1]), float(height_mm)])
    if float(tip[2]) < 40.0:
        return None
    pitch = float(pose[3] if pitch_deg is None else pitch_deg)
    target = (float(tip[0]), float(tip[1]), float(tip[2]), pitch)
    previous = bool(kinematics.ELBOW_UP)
    try:
        kinematics.ELBOW_UP = True
        try:
            joints = np.asarray(inverse(*target), dtype=float).reshape(4)
        except (ValueError, NotImplementedError):
            return None
    finally:
        kinematics.ELBOW_UP = previous
    limits = JOINT_LIMITS_DEG
    if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
        return None
    return target


def scale_from_views(pixel_a, pixel_b, joints_a, joints_b):
    """Pixels per millimetre from one sideways camera move.

    The two pictures must be of the same ring, and the camera must have
    slid across the image rather than along its view. Returns (scale, None)
    or (None, reason). A negative scale means an image axis is flipped.
    """
    origin_a, rotation_a = _camera_pose(joints_a)
    origin_b, rotation_b = _camera_pose(joints_b)
    axes = rotation_a @ CAMERA_AXES_IN_TOOL
    right, down, view = axes[:, 0], axes[:, 1], axes[:, 2]
    view_b = (rotation_b @ CAMERA_AXES_IN_TOOL)[:, 2]
    if float(np.dot(view, view_b)) < 0.94:
        return None, "the camera turned too far to reuse the pixel scale"
    moved = origin_b - origin_a
    travel_mm = float(np.linalg.norm(moved))
    if travel_mm < 1.0:
        return None, "the camera did not move"
    if abs(float(np.dot(moved, view))) > 0.35 * travel_mm:
        return None, "the camera moved along its view, so the scale is not usable"
    along = np.array([float(np.dot(moved, right)), float(np.dot(moved, down))])
    axis = int(np.argmax(np.abs(along)))
    travel = float(along[axis])
    if abs(travel) < 8.0:
        return None, "the camera barely moved across the image"
    delta = np.array(
        [float(pixel_b[0]) - float(pixel_a[0]), float(pixel_b[1]) - float(pixel_a[1])],
        dtype=float,
    )
    if abs(float(delta[axis])) < 12.0:
        return None, "the ring did not move in the image. Leave it still and run again."
    other = 1 - axis
    if abs(float(delta[other])) > max(15.0, 0.5 * abs(float(delta[axis]))):
        return None, "the ring jumped across the image. It has to be the same ring."
    scale = -float(delta[axis]) / travel
    if scale <= 0.0:
        name = "image-right" if axis == 0 else "image-down"
        return None, f"a negative scale means {name} is flipped in CAMERA_AXES_IN_TOOL"
    if not 0.4 <= scale <= 25.0:
        return None, f"pixel scale {scale:.2f} px/mm is not usable"
    return scale, None


def view_on_plane(joint_angles_deg, plane_z=None):
    """Where the center of the image meets the ring plane. None if it misses."""
    height = RING_PLANE_Z if plane_z is None else float(plane_z)
    origin, rotation = _camera_pose(joint_angles_deg)
    view = (rotation @ CAMERA_AXES_IN_TOOL)[:, 2]
    if abs(float(view[2])) < 1e-6:
        return None
    distance = (height - float(origin[2])) / float(view[2])
    if distance <= 1.0:
        return None
    return origin + distance * view


def table_from_scale(pixel, image_shape, joint_angles_deg, scale_px_per_mm, plane_z=None):
    """Ring on the plane from one pixel and a pixels-per-millimetre scale.

    scale_px_per_mm is f/Z at the ring, from scale_from_views. Returns
    ((x, y, z), None) or (None, reason).
    """
    scale = float(scale_px_per_mm)
    if scale <= 0.0:
        return None, "pixel scale must be positive"
    height = RING_PLANE_Z if plane_z is None else float(plane_z)
    origin, rotation = _camera_pose(joint_angles_deg)
    axes = rotation @ CAMERA_AXES_IN_TOOL
    right, down, view = axes[:, 0], axes[:, 1], axes[:, 2]
    cx = 0.5 * (int(image_shape[1]) - 1)
    cy = 0.5 * (int(image_shape[0]) - 1)
    offset = np.array(
        [(float(pixel[0]) - cx) / scale, (float(pixel[1]) - cy) / scale],
        dtype=float,
    )
    lateral = origin + float(offset[0]) * right + float(offset[1]) * down
    if abs(float(view[2])) < 1e-6:
        return None, "the camera is looking along the table"
    distance = (height - float(lateral[2])) / float(view[2])
    if distance <= 1.0:
        return None, "the ring ray does not meet the table in front of the camera"
    return lateral + distance * view, None


def step_toward_pixel(
    joint_angles_deg,
    pixel,
    image_shape,
    scale_px_per_mm,
    gain=0.6,
    max_step_mm=20.0,
    min_step_mm=4.0,
    pitch_deg=None,
    height_mm=None,
):
    """Slide the tip toward the ring in the image.

    pitch_deg locks the tool angle. None keeps the pitch the arm has now.
    height_mm locks the tip height. None follows the image step in z.

    The step is gain times the pixel offset divided by scale_px_per_mm,
    clipped to max_step_mm, with the along-view part removed so the arm
    does not drive into the table. Returns (x, y, z, pitch_deg), or None
    when the step is under min_step_mm or does not fit.
    """
    from robot.arm import JOINT_LIMITS_DEG
    from robot.kinematics import _lowest_link_z, forward, inverse
    from robot import kinematics

    scale = float(scale_px_per_mm)
    if scale <= 0.0:
        return None
    cx = 0.5 * (int(image_shape[1]) - 1)
    cy = 0.5 * (int(image_shape[0]) - 1)
    raw = np.array(
        [(float(pixel[0]) - cx) / scale, (float(pixel[1]) - cy) / scale],
        dtype=float,
    )
    length = float(np.linalg.norm(raw))
    if length < float(min_step_mm):
        return None
    step = float(gain) * raw
    step_length = float(np.linalg.norm(step))
    if step_length > float(max_step_mm):
        step *= float(max_step_mm) / step_length
    _origin, rotation = _camera_pose(joint_angles_deg)
    axes = rotation @ CAMERA_AXES_IN_TOOL
    delta = float(step[0]) * axes[:, 0] + float(step[1]) * axes[:, 1]
    view = axes[:, 2]
    delta = delta - float(np.dot(delta, view)) * view
    if height_mm is None and float(delta[2]) < -8.0:
        delta = delta.copy()
        delta[2] = -8.0
    if height_mm is not None:
        delta = np.array([float(delta[0]), float(delta[1]), 0.0])
    if float(np.linalg.norm(delta)) < float(min_step_mm):
        return None
    pose = forward(joint_angles_deg)
    tip = pose[:3] + delta
    if height_mm is not None:
        tip[2] = float(height_mm)
    if float(tip[2]) < 50.0:
        return None
    pitch = float(pose[3] if pitch_deg is None else pitch_deg)
    target = (float(tip[0]), float(tip[1]), float(tip[2]), pitch)
    previous = bool(kinematics.ELBOW_UP)
    try:
        kinematics.ELBOW_UP = True
        try:
            joints = np.asarray(inverse(*target), dtype=float).reshape(4)
        except (ValueError, NotImplementedError):
            return None
    finally:
        kinematics.ELBOW_UP = previous
    limits = JOINT_LIMITS_DEG
    if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
        return None
    if _lowest_link_z(joints) < 40.0:
        return None
    return target


def point_from_shift(pixel_a, pixel_b, image_shape, joints_a, joints_b, plane_z=None):
    """Where the ring is, from two pictures and the camera motion between them.

    The shift gives pixels per millimetre at the ring. The ring plane gives
    the distance along the view. No focal length and no measured (x, y).
    Returns ((x, y, z), None), or (None, reason) when the two pictures do
    not determine a point on the plane.
    """
    scale, note = scale_from_views(pixel_a, pixel_b, joints_a, joints_b)
    if scale is None:
        return None, note
    return table_from_scale(pixel_a, image_shape, joints_a, scale, plane_z)


def hover_over(point_xyz, standoff_mm=200.0, pitch_deg=None):
    """Camera pose standoff_mm from point_xyz, as steep as the arm can hold.

    Straight above the ring at this distance is out of reach once the ring
    is away from the base. The view still passes through the point.
    pitch_deg, when given, is tried first so a tracking step does not
    change the aim angle.
    """
    if pitch_deg is not None:
        pose = aim_camera_at(point_xyz, pitch_deg, standoff_mm)
        if pose is not None:
            return pose
    for pitch in (-55.0, -50.0, -45.0, -40.0, -35.0, -30.0):
        pose = aim_camera_at(point_xyz, pitch, standoff_mm)
        if pose is not None:
            return pose
    for standoff in (180.0, 160.0, 220.0):
        if abs(standoff - float(standoff_mm)) < 1.0:
            continue
        for pitch in (-50.0, -40.0, -30.0):
            pose = aim_camera_at(point_xyz, pitch, standoff)
            if pose is not None:
                return pose
    return None


def pixel_to_point(pixel, image_shape):
    """Map a camera pixel to a point the arm can aim at.

    pixel is (u, v). image_shape is the frame's (height, width) or
    (height, width, channels). Return (x, y, z, pitch_deg) in the robot frame.

    This needs the camera mount and its pixel-to-direction scale. Until that
    calibration is filled in, pointing at a ring has nowhere to aim.
    """
    raise NotImplementedError(
        "Add the camera mounting geometry in robot/calibration.py. "
        "pixel_to_point should return (x, y, z, pitch_deg)."
    )
