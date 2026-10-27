import numpy as np

from robot.kinematics import inverse


def follow_line(start_pose, end_pose, duration_s, dt=0.1):
    """Sample a straight Cartesian line into timed joint angles.

    start_pose and end_pose are (x, y, z, pitch_deg).
    Each sample is (time_s, [j1, j2, j3, j4]).
    This calls inverse() and does not move the arm.
    """
    start = np.asarray(start_pose, dtype=float).reshape(4)
    end = np.asarray(end_pose, dtype=float).reshape(4)
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    n = max(1, int(round(duration_s / dt)))
    samples = []
    for i in range(n + 1):
        alpha = i / n
        pose = (1.0 - alpha) * start + alpha * end
        joints = np.asarray(inverse(pose[0], pose[1], pose[2], pose[3]), dtype=float).reshape(4)
        samples.append((alpha * duration_s, joints))
    return samples


def joint_trajectory(waypoints, duration_s):
    """Interpolate recorded joint waypoints. No inverse kinematics.

    waypoints is an (N, 4) array of degrees.
    Each sample is (time_s, [j1, j2, j3, j4]), one per waypoint,
    spread evenly across duration_s.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 4:
        raise ValueError("waypoints must have shape (N, 4)")
    if len(waypoints) == 0:
        return []
    if duration_s < 0:
        raise ValueError("duration_s must be >= 0")

    if len(waypoints) == 1:
        times = [0.0]
    else:
        times = np.linspace(0.0, duration_s, len(waypoints))
    return [(float(t), waypoints[i].copy()) for i, t in enumerate(times)]
