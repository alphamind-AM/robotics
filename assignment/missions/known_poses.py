"""Poses with a simple shape, so the real arm can be compared with the model.

Run it on the bench first, then on the arm. Each pose is held for a few
seconds. The terminal prints what the model claims.

    python MyRobot.py --check --mission known_poses
    python MyRobot.py --view --mission known_poses
    python MyRobot.py --mission known_poses

Clear the space in front of the arm. Joint 1 can only yaw ±130°, so the
forearm sweeps across the front between those stops. It cannot point
straight back.

The model frame has z up and x forward. y completes a right-handed frame,
so with z up and x forward, +y is to the left. Lengths are LINK_1..LINK_4
in robot/kinematics.py. A joint angle of 0° is the model's zero, which is
straight up for the shoulder and "continue the previous link" for the
elbow and wrist.

If the upright pose is not a vertical line, a joint zero does not match
the servo. If that pose is vertical but the horizontal forearm points the
wrong way, the base yaw or its sign is wrong. The later moves use inverse,
so they are only worth measuring after the joint poses already match.
"""

from robot.arm import JOINT_LIMITS_DEG
from robot.kinematics import LINK_1, LINK_2, LINK_3, LINK_4, forward
from robot.mission import Mission, go, joints

HOLD_S = 5.0

# Straight up: base, shoulder, elbow, wrist, tip, all on the z axis.
TIP_Z = float(LINK_1 + LINK_2 + LINK_3 + LINK_4)
# Upper arm vertical, forearm and tool horizontal.
ELBOW_Z = float(LINK_1 + LINK_2)
FOREARM = float(LINK_3 + LINK_4)

# Level points used by inverse. Same height, same distance from the base
# axis, tool horizontal. Pitch 0 is horizontal along the reach.
LEVEL_Z = 160.0
LEVEL_REACH = 160.0
PITCH = 0.0

# Joint 1 stops at ±130°. Stay 5° inside so a goal is not the physical stop.
J1_NEAR_STOP = float(JOINT_LIMITS_DEG[0, 1]) - 5.0


def _level(x, y):
    return (float(x), float(y), LEVEL_Z, PITCH)


def _yaw_note(j1_deg, sentence):
    pose = forward((j1_deg, 0.0, 90.0, 0.0))
    return (
        f"{sentence} Tip at x={pose[0]:.0f}, y={pose[1]:.0f}, z={pose[2]:.0f} mm."
    )


known_poses = Mission(
    "known_poses",
    [
        joints(
            (0.0, 0.0, 0.0, 0.0),
            "upright",
            HOLD_S,
            f"Vertical line on the base axis. Tip at x=0, y=0, z={TIP_Z:.0f} mm. Tool points up.",
        ),
        joints(
            (0.0, 0.0, 0.0, 90.0),
            "wrist toward +x",
            HOLD_S,
            f"Arm stays vertical. Tool points along +x. Tip at x={LINK_4:.0f}, y=0, z={ELBOW_Z + LINK_3:.0f} mm.",
        ),
        joints(
            (0.0, 0.0, 90.0, 0.0),
            "forearm +x",
            HOLD_S,
            f"Upper arm vertical. Forearm horizontal along +x at z={ELBOW_Z:.0f} mm. "
            f"Tip at x={FOREARM:.0f}, y=0.",
        ),
        joints(
            (90.0, 0.0, 90.0, 0.0),
            "forearm +y",
            HOLD_S,
            f"Same shape along +y, to the left of +x. Tip at x=0, y={FOREARM:.0f}, z={ELBOW_Z:.0f} mm.",
        ),
        joints(
            (J1_NEAR_STOP, 0.0, 90.0, 0.0),
            "yaw near +130",
            HOLD_S,
            _yaw_note(J1_NEAR_STOP, "Joint 1 near its +130° stop. The base cannot reach 180°."),
        ),
        joints(
            (-J1_NEAR_STOP, 0.0, 90.0, 0.0),
            "yaw near -130",
            HOLD_S,
            _yaw_note(-J1_NEAR_STOP, "Joint 1 near its -130° stop, sweeping across the front."),
        ),
        joints(
            (-90.0, 0.0, 90.0, 0.0),
            "forearm -y",
            HOLD_S,
            f"Same shape along -y. Tip at x=0, y={-FOREARM:.0f}, z={ELBOW_Z:.0f} mm.",
        ),
        go(
            _level(LEVEL_REACH, 0.0),
            "inverse +x",
            elbow_up=True,
            settle_s=HOLD_S,
            note=f"Inverse. Tip at x={LEVEL_REACH:.0f}, y=0, z={LEVEL_Z:.0f} mm, tool horizontal.",
        ),
        go(
            _level(0.0, LEVEL_REACH),
            "inverse +y",
            elbow_up=True,
            settle_s=HOLD_S,
            note=f"Inverse. Tip at x=0, y={LEVEL_REACH:.0f}, z={LEVEL_Z:.0f} mm, tool horizontal.",
        ),
    ],
)
