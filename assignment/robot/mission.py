"""A mission is a list of steps. Each step calls the arm. It does not call inverse.

Poses are (x, y, z, pitch_deg). Shapes in missions/shapes.py build steps
with go() and slide(). A mission file only puts those steps in order.
joints() commands joint angles directly, which is what a zero check needs.
"""

import time

import numpy as np

from robot.behaviors import follow_cartesian_line
from robot.kinematics import forward, inverse, jacobian


class Mission:
    def __init__(self, name, steps):
        self.name = name
        self.steps = list(steps)

    def run(self, robot, context=None, announce=None):
        context = {} if context is None else context
        for index, step in enumerate(self.steps):
            if announce is not None:
                announce(index, step)
            step(robot, context)


def _with_elbow(elbow_up, fn):
    """Run fn with one elbow branch. None leaves the global choice alone."""
    if elbow_up is None:
        return fn()
    from robot import kinematics

    previous = bool(kinematics.ELBOW_UP)
    kinematics.ELBOW_UP = bool(elbow_up)
    try:
        return fn()
    finally:
        kinematics.ELBOW_UP = previous


def _settle(robot, seconds):
    """Wait on the real arm so a pose can be measured. The preview does not wait."""
    if seconds <= 0 or getattr(robot, "bus", None) is None:
        return
    time.sleep(float(seconds))


def _say(note):
    if note:
        print(f"    {note}")


def go(pose, label="go", elbow_up=None, settle_s=0.0, note=""):
    """Move to one pose. The arm takes whatever joint path move_c produces.

    elbow_up selects the planar branch for this step only. Leave it out to
    use the global ELBOW_UP flag. settle_s holds the real arm at the pose.
    """
    pose = tuple(float(v) for v in pose)

    def step(robot, context):
        _say(note)
        _with_elbow(elbow_up, lambda: robot.move_c(*pose))
        _settle(robot, settle_s)

    step.label = label
    return step


def joints(angles, label="joints", settle_s=0.0, note=""):
    """Move to joint angles in degrees. Does not call inverse.

    This is the check for a joint zero or a joint sign. The commanded
    angles are the angles, whatever pose inverse would have chosen.
    """
    angles = tuple(float(v) for v in angles)

    def step(robot, context):
        _say(note)
        robot.move_j(*angles)
        _settle(robot, settle_s)

    step.label = label
    return step


def slide(start, end, duration_s=2.0, label="slide", elbow_up=None):
    """Move in a straight line from start to end. Both are poses."""
    start = tuple(float(v) for v in start)
    end = tuple(float(v) for v in end)

    def step(robot, context):
        _with_elbow(
            elbow_up,
            lambda: follow_cartesian_line(robot, start, end, duration_s),
        )

    step.label = label
    return step


def decide(choose, label="decide"):
    """Run choose(robot, context) when the mission reaches this step.

    choose writes context["pose"] as (x, y, z, pitch_deg). The pose is
    chosen at run time. go() cannot do that: its pose is fixed when the
    mission file is loaded.
    """

    def step(robot, context):
        choose(robot, context)

    step.label = label
    return step


def go_chosen(key="pose", label="go", elbow_up=None, settle_s=0.0):
    """Move to the pose a decide step stored in context[key].

    No pose means the arm is not commanded. That is the miss case: the
    decision found nothing, so there is nowhere to go. settle_s holds the
    real arm at the pose.
    """

    def step(robot, context):
        pose = context.get(key)
        if pose is None:
            print("    no pose, so the arm stays")
            return
        pose = tuple(float(v) for v in pose)
        context[key] = pose
        _with_elbow(elbow_up, lambda: robot.move_c(*pose))
        _settle(robot, settle_s)

    step.label = label
    return step


def _pose_text(pose):
    pose = np.asarray(pose, dtype=float).reshape(4)
    return (
        f"x={pose[0]:.1f}  y={pose[1]:.1f}  z={pose[2]:.1f}  "
        f"pitch={pose[3]:.1f}"
    )


def _joint_text(joints):
    joints = np.asarray(joints, dtype=float).reshape(4)
    return "  ".join(f"j{i + 1}={angle:.1f}" for i, angle in enumerate(joints))


def _sample_joints(sample):
    if isinstance(sample, (tuple, list)) and len(sample) == 2 and np.isscalar(sample[0]):
        return float(sample[0]), np.asarray(sample[1], dtype=float).reshape(4)
    return None, np.asarray(sample, dtype=float).reshape(4)


def _wrapped_delta_deg(start, end):
    """Shortest step from start to end, in degrees, one value per joint."""
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return (delta + 180.0) % 360.0 - 180.0


def _joint_rates(samples):
    """Forward-difference joint rate in deg/s. The last sample keeps the previous rate."""
    parsed = [_sample_joints(sample) for sample in samples]
    rates = [None] * len(parsed)
    for i in range(len(parsed) - 1):
        t0, q0 = parsed[i]
        t1, q1 = parsed[i + 1]
        if t0 is None or t1 is None or t1 <= t0:
            continue
        rates[i] = _wrapped_delta_deg(q0, q1) / (t1 - t0)
    if len(rates) >= 2 and rates[-2] is not None:
        rates[-1] = rates[-2]
    return parsed, rates


def _print_state(joints, qdot_deg_s=None, time_s=None):
    """Joints, the forward pose, and the twist J(q) qdot when a rate exists."""
    joints = np.asarray(joints, dtype=float).reshape(4)
    clock = f"t={time_s:5.2f}  " if time_s is not None else ""
    print(f"    {clock}joints {_joint_text(joints)}")
    print(f"    {_pose_text(forward(joints))}")
    if qdot_deg_s is None:
        print("    velocity: no sample time")
        return
    qdot_deg_s = np.asarray(qdot_deg_s, dtype=float).reshape(4)
    twist = jacobian(joints) @ np.deg2rad(qdot_deg_s)
    linear = "  ".join(f"v{axis}={twist[i]:7.1f}" for i, axis in enumerate("xyz"))
    angular = "  ".join(
        f"w{axis}={np.rad2deg(twist[3 + i]):7.1f}" for i, axis in enumerate("xyz")
    )
    print(f"    qdot {_joint_text(qdot_deg_s)} deg/s")
    print(f"    {linear} mm/s")
    print(f"    {angular} deg/s")


class PreviewRobot:
    """Records the poses a mission would send. Does not open the robot."""

    def move_c(self, x, y, z, pitch_deg, **kwargs):
        joints = inverse(x, y, z, pitch_deg)
        _print_state(joints)

    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, **kwargs):
        _print_state((j1_deg, j2_deg, j3_deg, j4_deg))

    def execute(self, samples):
        if len(samples) == 0:
            print("    line has no samples")
            return
        parsed, rates = _joint_rates(samples)
        print(f"    {len(parsed)} samples")
        for (time_s, joints), qdot in zip(parsed, rates):
            _print_state(joints, qdot, time_s)


def preview_mission(mission, context=None):
    """Print each step. Does not connect to the robot.

    context is the same dict the steps see. A pose put there before run
    is the runtime target of go_chosen().
    """
    print(f"Mission {mission.name}. The robot is not connected.")
    print("Velocity is J(q) qdot on timed samples. A go() step has no rate.")
    robot = PreviewRobot()

    def announce(index, step):
        print(f"{index + 1}. {getattr(step, 'label', 'step')}")

    try:
        mission.run(robot, context=context, announce=announce)
    except (ValueError, NotImplementedError) as exc:
        print(f"    stopped: {exc}")
        return False
    print(f"{len(mission.steps)} steps")
    return True
