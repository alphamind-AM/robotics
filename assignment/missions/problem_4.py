"""Jacobians of the tip and the camera along the Problem 3 circle.

The Jacobian itself is a function of the joint angles q. Each angle phi on
the circle is a pose P. inverse(P) supplies the q that jacobian(q) needs.
"""

import numpy as np

from missions.problem_3 import circle_pose
from robot.kinematics import inverse, jacobian, jacobian_camera


PHIS = (
    ("0", 0.0),
    ("pi/2", np.pi / 2),
    ("pi", np.pi),
    ("3pi/2", 3 * np.pi / 2),
)


def jacobians_at_pose(pose):
    """Pose P = (x, y, z, pitch_deg) -> (q, J_tip, J_camera).

    q is in degrees. Both Jacobians are 6x4 and expect joint rates in rad/s.
    """
    q = np.asarray(inverse(*pose), dtype=float).reshape(4)
    return q, jacobian(q), jacobian_camera(q)


def _fmt_matrix(matrix):
    return np.array2string(
        np.asarray(matrix, dtype=float),
        precision=3,
        suppress_small=True,
        floatmode="fixed",
    )


def report():
    """Print both Jacobians at phi = 0, pi/2, pi, and 3pi/2."""
    print("Problem 4. Joint rates that multiply these matrices are in rad/s.")
    print("Top three rows: mm/s. Bottom three rows: rad/s.")
    for name, phi in PHIS:
        pose = circle_pose(phi)
        q, jac_tip, jac_camera = jacobians_at_pose(pose)
        x, y, z, pitch_deg = pose
        print()
        print(f"phi = {name}")
        print(f"P = ({x:.3f}, {y:.3f}, {z:.3f}, pitch {pitch_deg:.1f} deg)")
        print(
            "q = "
            + "  ".join(f"j{i + 1}={angle:.3f}" for i, angle in enumerate(q))
            + " deg"
        )
        print("J end effector =")
        print(_fmt_matrix(jac_tip))
        print("J camera =")
        print(_fmt_matrix(jac_camera))


if __name__ == "__main__":
    report()
