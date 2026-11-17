"""Look at the table and keep moving until the ring sits near the image center.

The arm does not touch the ring. It learns pixels per millimetre from one
short camera slide, then creeps toward the ring a few millimetres at a
time. The tool stays at the look height for the whole search. The camera
window stays open the whole time, including while the arm is moving. q in
that window stops the loop and leaves the arm where it is. Slide the ring
and the arm follows.

The ring's height is the table plane, RING_PLANE_Z in robot/calibration.py.
Every commanded pose keeps the tool at pitch -90 degrees, so the camera looks down.

    python MyRobot.py --mission point_ring

`--pose X Y Z PITCH` skips the camera and goes to that pose once.
`--check` and `--view` do not open the arm or the camera. They follow a
stand-in ring offset from the look pose.
"""

import time

import numpy as np

from robot.calibration import (
    CAMERA_AXES_IN_TOOL,
    _camera_pose,
    scale_from_views,
    shift_pose,
    step_toward_pixel,
    table_from_scale,
    view_on_plane,
)
from robot.kinematics import forward, inverse
from robot.mission import Mission, _settle, _with_elbow, decide, joints
from robot.perception import detect_rings

HOLD_S = 2.0
ARM_CAMERA = "/dev/video4"
# The look the arm starts from. Pitch is -90°, camera straight down.
# Joint 4 at 115° stays inside its ±140° stop.
LOOK_JOINTS = (0.0, -20.0, 85.0, 115.0)
DOWN_PITCH = -90.0
# Tip height of LOOK_JOINTS. The slide and the tracking steps stay here.
LOOK_HEIGHT_MM = float(forward(LOOK_JOINTS)[2])
SHIFT_MM = 18.0
# Stop correcting once the ring is this close to the image center, and
# start again only after it drifts out to RELEASE_PX.
CENTER_PX = 30.0
RELEASE_PX = 50.0
# Fraction of the pixel error in one command. Commands are a few millimetres,
# sent again as soon as the arm is near the previous one.
GAIN = 0.5
MAX_STEP_MM = 2.0
MIN_STEP_MM = 0.8
COMMAND_S = 0.06
_BENCH_FOCAL = 700.0


def _project(joint_angles_deg, point, focal_px, shape=(480, 640, 3)):
    origin, rotation = _camera_pose(joint_angles_deg)
    optical = CAMERA_AXES_IN_TOOL.T @ (rotation.T @ (np.asarray(point, dtype=float) - origin))
    if float(optical[2]) <= 1.0:
        return None
    cx = 0.5 * (int(shape[1]) - 1)
    cy = 0.5 * (int(shape[0]) - 1)
    return (
        cx + float(focal_px) * float(optical[0]) / float(optical[2]),
        cy + float(focal_px) * float(optical[1]) / float(optical[2]),
    )


def _pixel_error(pixel, shape):
    cx = 0.5 * (int(shape[1]) - 1)
    cy = 0.5 * (int(shape[0]) - 1)
    return float(np.hypot(float(pixel[0]) - cx, float(pixel[1]) - cy))


def _go(robot, pose, wait=True):
    """Elbow-up cartesian move. wait=False returns as soon as the goal is sent."""
    goal = {}

    def run():
        goal["joints"] = np.asarray(inverse(*pose), dtype=float).reshape(4)
        robot.move_c(*pose, wait=wait)

    _with_elbow(True, run)
    if wait:
        _settle(robot, 0.3)
    return goal.get("joints")


def _arrived(robot, goal, tol_deg=3.0):
    """True when the arm is near goal, or when this robot cannot report joints."""
    if goal is None or not hasattr(robot, "read_joint_angles"):
        return True
    now = np.asarray(robot.read_joint_angles(), dtype=float).reshape(4)
    delta = (now - np.asarray(goal, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta))) < float(tol_deg)


def _stable(phase, rings):
    """Mean of the last centers once they agree, else None."""
    recent = phase["recent"]
    if not rings:
        recent.clear()
        return None
    recent.append((float(rings[0][0]), float(rings[0][1])))
    if len(recent) > 12:
        del recent[:-12]
    if len(recent) < 8:
        return None
    cloud = np.asarray(recent, dtype=float)
    if np.linalg.norm(cloud - cloud.mean(axis=0), axis=1).max() > 20.0:
        return None
    center = cloud.mean(axis=0)
    return float(center[0]), float(center[1])


def _track_pose(pixel, shape, joints, scale, pitch, height_mm):
    """One partial step toward the ring, at a fixed pitch and tip height."""
    pose = step_toward_pixel(
        joints,
        pixel,
        shape,
        scale,
        gain=GAIN,
        max_step_mm=MAX_STEP_MM,
        min_step_mm=MIN_STEP_MM,
        pitch_deg=pitch,
        height_mm=height_mm,
    )
    if pose is None:
        return None, "hold"
    return pose, None


def _smooth(phase, rings):
    """Short average of the latest ring. A few missed frames keep the last one."""
    if not rings:
        phase["miss"] = phase.get("miss", 0) + 1
        if phase["miss"] > 6:
            phase["pixel"] = None
        return phase.get("pixel")
    phase["miss"] = 0
    fresh = np.array([float(rings[0][0]), float(rings[0][1])], dtype=float)
    previous = phase.get("pixel")
    if previous is None:
        phase["pixel"] = fresh
    else:
        phase["pixel"] = 0.45 * np.asarray(previous, dtype=float) + 0.55 * fresh
    smoothed = phase["pixel"]
    return float(smoothed[0]), float(smoothed[1])


def _live_step(robot, phase, frame, rings):
    """One camera frame. None keeps the same window open."""
    shape = frame.shape
    if phase["name"] == "see":
        pixel = _stable(phase, rings)
        if pixel is None:
            return None
        joints = np.asarray(robot.read_joint_angles(), dtype=float).reshape(4)
        pose = shift_pose(
            joints, SHIFT_MM, pitch_deg=DOWN_PITCH, height_mm=LOOK_HEIGHT_MM
        )
        if pose is None:
            print("    the camera slide does not fit, so the arm stays")
            return "stop"
        print(f"    ring pixel u={pixel[0]:.0f}  v={pixel[1]:.0f}")
        print(
            f"    sliding the camera to x={pose[0]:.0f}  y={pose[1]:.0f}  "
            f"z={pose[2]:.0f}  pitch={pose[3]:.0f}"
        )
        phase["sight"] = (pixel, shape, joints.copy())
        phase["goal"] = _go(robot, pose, wait=False)
        phase["name"] = "sliding"
        phase["recent"] = []
        return None

    if phase["name"] == "sliding":
        if not _arrived(robot, phase.get("goal")):
            return None
        phase["recent"] = []
        phase["name"] = "scale"
        return None

    if phase["name"] == "scale":
        pixel = _stable(phase, rings)
        if pixel is None:
            return None
        joints = np.asarray(robot.read_joint_angles(), dtype=float).reshape(4)
        pixel_a, shape_a, joints_a = phase["sight"]
        scale, note = scale_from_views(pixel_a, pixel, joints_a, joints)
        if scale is None:
            print(f"    {note}")
            return "stop"
        point, note = table_from_scale(pixel_a, shape_a, joints_a, scale)
        if point is None:
            print(f"    {note}")
            return "stop"
        phase["scale"] = scale
        phase["holding"] = False
        phase["pixel"] = None
        phase["sent"] = 0.0
        phase["told"] = 0.0
        print(f"    ring on the table x={point[0]:.0f}  y={point[1]:.0f}  z={point[2]:.0f}")
        print(f"    pixel scale {scale:.2f} px/mm")
        print("    following the ring. q stops.")
        phase["name"] = "track"
        return None

    pixel = _smooth(phase, rings)
    if pixel is None or phase.get("scale") is None:
        return None
    now = time.monotonic()
    if now - phase.get("sent", 0.0) < COMMAND_S:
        return None
    if not _arrived(robot, phase.get("goal")):
        return None
    error = _pixel_error(pixel, shape)
    if phase.get("holding") and error <= RELEASE_PX:
        return None
    if error <= CENTER_PX:
        if not phase.get("holding"):
            print(f"    pixel error {error:.0f} px. Near the center. q stops, or move the ring.")
        phase["holding"] = True
        return None
    phase["holding"] = False
    joints = np.asarray(robot.read_joint_angles(), dtype=float).reshape(4)
    pose, _note = _track_pose(
        pixel, shape, joints, phase["scale"], DOWN_PITCH, LOOK_HEIGHT_MM
    )
    if pose is None:
        return None
    phase["goal"] = _go(robot, pose, wait=False)
    phase["sent"] = now
    if now - phase.get("told", 0.0) > 0.5:
        print(
            f"    pixel error {error:.0f} px  x={pose[0]:.0f}  y={pose[1]:.0f}  "
            f"z={pose[2]:.0f}  pitch={pose[3]:.0f}"
        )
        phase["told"] = now
    return None


def _follow_live(robot, camera, device):
    phase = {"name": "see", "recent": [], "scale": None, "goal": None}

    def consider(frame, rings):
        try:
            return _live_step(robot, phase, frame, rings)
        except ValueError as exc:
            print(f"    {exc}")
            return "stop"

    camera.watch_until(detect_rings, device, consider)


def _bench_follow(robot):
    """Same loop as the arm, with a ring drawn into a pinhole at focal 700."""
    hit = view_on_plane(LOOK_JOINTS)
    if hit is None:
        print("    bench look does not see the table")
        return
    ring = np.array([float(hit[0]), float(hit[1]) + 30.0, float(hit[2])], dtype=float)
    shape = (480, 640, 3)
    joints = np.asarray(LOOK_JOINTS, dtype=float)

    def joints_of(pose):
        from robot import kinematics

        previous = bool(kinematics.ELBOW_UP)
        kinematics.ELBOW_UP = True
        try:
            return np.asarray(inverse(*pose), dtype=float).reshape(4)
        finally:
            kinematics.ELBOW_UP = previous

    pixel = _project(joints, ring, _BENCH_FOCAL, shape)
    if pixel is None:
        print("    bench ring is outside the picture")
        return
    print(f"    no camera. Bench ring at x={ring[0]:.0f}  y={ring[1]:.0f}.")
    pose = shift_pose(joints, SHIFT_MM, pitch_deg=DOWN_PITCH, height_mm=LOOK_HEIGHT_MM)
    if pose is None:
        print("    bench slide is out of reach")
        return
    print(
        f"    sliding the camera to x={pose[0]:.0f}  y={pose[1]:.0f}  "
        f"z={pose[2]:.0f}  pitch={pose[3]:.0f}"
    )
    _go(robot, pose)
    joints_b = joints_of(pose)
    pixel_b = _project(joints_b, ring, _BENCH_FOCAL, shape)
    if pixel_b is None:
        print("    bench ring left the picture")
        return
    scale, note = scale_from_views(pixel, pixel_b, joints, joints_b)
    if scale is None:
        print(f"    {note}")
        return
    point, note = table_from_scale(pixel, shape, joints, scale)
    if point is None:
        print(f"    {note}")
        return
    error_mm = float(np.linalg.norm(np.asarray(point) - ring))
    print(f"    bench ring error {error_mm:.1f} mm  scale {scale:.2f} px/mm")
    print("    following the ring in small steps")
    joints = joints_b
    for move in range(1, 81):
        pixel = _project(joints, ring, _BENCH_FOCAL, shape)
        if pixel is None:
            print("    bench ring left the picture")
            return
        error = _pixel_error(pixel, shape)
        if error <= CENTER_PX:
            print(f"    pixel error {error:.0f} px after {move - 1} steps. Near the center.")
            return
        pose, note = _track_pose(pixel, shape, joints, scale, DOWN_PITCH, LOOK_HEIGHT_MM)
        if pose is None:
            print(f"    pixel error {error:.0f} px. {note or 'no step'}")
            return
        if move == 1 or move % 8 == 0:
            print(
                f"    step {move}  pixel error {error:.0f} px  "
                f"x={pose[0]:.0f}  y={pose[1]:.0f}  z={pose[2]:.0f}"
            )
        _go(robot, pose)
        joints = joints_of(pose)
    print("    stopped after 80 steps")


def follow_ring(robot, context):
    """Slide, hover, then correct while the camera stays open."""
    if context.get("pose") is not None:
        pose = list(context["pose"])
        pose[3] = -90.0
        print(
            f"    pose already set, x={pose[0]:.1f}  y={pose[1]:.1f}  "
            f"z={pose[2]:.1f}  pitch={pose[3]:.1f}"
        )
        _go(robot, pose)
        return

    camera = getattr(robot, "camera", None)
    if camera is None or not hasattr(robot, "read_joint_angles"):
        _bench_follow(robot)
        return
    device = context.get("camera_device") or ARM_CAMERA
    _follow_live(robot, camera, device)


point_ring = Mission(
    "point_ring",
    [
        joints(
            LOOK_JOINTS,
            "look at the table",
            HOLD_S,
            "Camera looks down at the table in front of the base. Place one ring there.",
        ),
        decide(follow_ring, "follow the ring"),
    ],
)
