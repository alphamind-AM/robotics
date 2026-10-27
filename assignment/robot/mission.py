"""A mission is a list of steps. Each step calls the arm. It does not call inverse.

Poses are (x, y, z, pitch_deg). Shapes in missions/shapes.py build steps
with go() and slide(). A mission file only puts those steps in order.
"""

import numpy as np

from robot.behaviors import follow_cartesian_line
from robot.kinematics import forward, inverse


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


def go(pose, label="go"):
    """Move to one pose. The arm takes whatever joint path move_c produces."""
    pose = tuple(float(v) for v in pose)

    def step(robot, context):
        robot.move_c(*pose)

    step.label = label
    return step


def slide(start, end, duration_s=2.0, label="slide"):
    """Move in a straight line from start to end. Both are poses."""
    start = tuple(float(v) for v in start)
    end = tuple(float(v) for v in end)

    def step(robot, context):
        follow_cartesian_line(robot, start, end, duration_s)

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


class PreviewRobot:
    """Records the poses a mission would send. Does not open the robot."""

    def move_c(self, x, y, z, pitch_deg, **kwargs):
        pose = (x, y, z, pitch_deg)
        joints = inverse(*pose)
        print(f"    {_pose_text(pose)}")
        print(f"    joints {_joint_text(joints)}")

    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, **kwargs):
        joints = (j1_deg, j2_deg, j3_deg, j4_deg)
        print(f"    joints {_joint_text(joints)}")
        print(f"    {_pose_text(forward(joints))}")

    def execute(self, samples):
        if len(samples) == 0:
            print("    line has no samples")
            return
        start = np.asarray(samples[0][1], dtype=float).reshape(4)
        end = np.asarray(samples[-1][1], dtype=float).reshape(4)
        print(f"    {len(samples)} samples")
        print(f"    from {_pose_text(forward(start))}")
        print(f"    to   {_pose_text(forward(end))}")


def preview_mission(mission):
    """Print each step. Does not connect to the robot."""
    print(f"Mission {mission.name}. The robot is not connected.")
    robot = PreviewRobot()

    def announce(index, step):
        print(f"{index + 1}. {getattr(step, 'label', 'step')}")

    try:
        mission.run(robot, announce=announce)
    except ValueError as exc:
        print(f"    stopped: {exc}")
        return False
    print(f"{len(mission.steps)} steps")
    return True
