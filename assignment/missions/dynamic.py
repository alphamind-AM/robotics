"""A mission whose target is chosen when it runs.

The first move is a fixed start. The second move is context["pose"], which
is filled in at run time. --pose on the command line supplies that target.
With no --pose, DEFAULT_POSE is used. Neither path connects to the robot
when --check or --view is set.

    python MyRobot.py --check --mission dynamic
    python MyRobot.py --view --mission dynamic
    python MyRobot.py --view --mission dynamic --pose 150 40 150 0

Left and right step through the commands in the view. Space plays them.
The pixel-to-robot conversion is not in this mission. choose_pose is the
step that will later write a pose from the camera.
"""

from robot.mission import Mission, decide, go, go_chosen

START = (150.0, 0.0, 120.0, 0.0)
DEFAULT_POSE = (150.0, 40.0, 150.0, 0.0)


def choose_pose(robot, context):
    """Keep a pose already supplied, otherwise use the stand-in target."""
    del robot
    if context.get("pose") is None:
        context["pose"] = DEFAULT_POSE
    x, y, z, pitch = context["pose"]
    print(f"    chosen pose x={x:.1f}  y={y:.1f}  z={z:.1f}  pitch={pitch:.1f}")


dynamic = Mission(
    "dynamic",
    [
        go(START, "start", elbow_up=True),
        decide(choose_pose, "choose pose"),
        go_chosen("pose", "go to chosen pose", elbow_up=True),
    ],
)
