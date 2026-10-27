from robot.calibration import pixel_to_point
from robot.kinematics import inverse
from robot.perception import detect_faces
from robot.trajectory import follow_line, joint_trajectory


def follow_cartesian_line(robot, start_pose, end_pose, duration_s=4.0, dt=0.1):
    """Move along a straight line in position. Requires inverse()."""
    samples = follow_line(start_pose, end_pose, duration_s, dt=dt)
    robot.execute(samples)
    return samples


def follow_joint_waypoints(robot, waypoints, duration_s):
    """Move through joint-space waypoints. Does not use inverse kinematics."""
    samples = joint_trajectory(waypoints, duration_s)
    robot.execute(samples)
    return samples


def point_at_faces(robot, camera):
    """Aim the arm at the largest face in one camera frame.

    Wire-up for later: frame -> face box -> robot point -> inverse -> move_j.
    """
    frame = camera.read_frame()
    faces = detect_faces(frame)
    if not faces:
        return None
    u, v, w, h = faces[0]
    center = (u + w / 2.0, v + h / 2.0)
    x, y, z, pitch_deg = pixel_to_point(center, frame.shape)
    joints = inverse(x, y, z, pitch_deg)
    robot.move_j(*joints)
    return faces[0]
