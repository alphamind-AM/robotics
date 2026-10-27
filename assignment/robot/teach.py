import json
import os
import select
import sys
import termios
import time
import tty

import numpy as np

ASSIGNMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACH_LIMITS_PATH = os.path.join(ASSIGNMENT_DIR, "teach_limits.json")
TEACH_HISTORY_PATH = os.path.join(ASSIGNMENT_DIR, "teach_history.json")


def record_configuration(robot):
    entry = {
        "j1": float(robot.joint_angles[0]),
        "j2": float(robot.joint_angles[1]),
        "j3": float(robot.joint_angles[2]),
        "j4": float(robot.joint_angles[3]),
        "speed": float(robot.motor_speed[0]),
        "torque": float(robot.motor_torque[0]),
        "gripper_open": bool(robot.gripper_open_flag),
    }
    robot.movement_history.append(entry)
    print(
        f"Recorded Speed: {entry['speed']:.3f}, Torque: {entry['torque']:.3f}\n"
        f"Joint Positions: {entry['j1']:.2f}, {entry['j2']:.2f}, {entry['j3']:.2f}, {entry['j4']:.2f}\n"
        f"Gripper open: {int(entry['gripper_open'])}"
    )
    return entry


def delete_last_recorded_configuration(robot):
    if not robot.movement_history:
        print("No last history position")
        return
    robot.movement_history.pop()


def record_measured_configuration(robot):
    """Store the pose the arm is actually in, for later playback."""
    robot.joint_angles = robot.read_joint_angles()
    return record_configuration(robot)


def save_movement_history(robot, path=TEACH_HISTORY_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(robot.movement_history, f, indent=2)
    print(f"Saved {len(robot.movement_history)} pose(s) to {path}")


def load_movement_history(robot, path=TEACH_HISTORY_PATH):
    with open(path, encoding="utf-8") as f:
        robot.movement_history = json.load(f)
    print(f"Loaded {len(robot.movement_history)} pose(s) from {path}")
    return robot.movement_history


def save_joint_limits(robot, joint_min, joint_max, path=TEACH_LIMITS_PATH):
    payload = {
        "min_deg": [float(v) for v in joint_min],
        "max_deg": [float(v) for v in joint_max],
        "joint_limits": [[float(joint_min[i]), float(joint_max[i])] for i in range(4)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved joint extremes to {path}")
    return payload


def load_joint_limits(robot, path=TEACH_LIMITS_PATH):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    robot.joint_limits = np.array(payload["joint_limits"], dtype=float)
    print("Joint limits set to the recorded extremes:")
    for i, (lo, hi) in enumerate(robot.joint_limits):
        print(f"  J{i + 1}: [{lo:.2f}, {hi:.2f}] deg")
    return robot.joint_limits


def play_configuration_history(robot, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
    if not robot.movement_history:
        print("No recorded configurations.")
        return
    for rec in robot.movement_history:
        s = rec["speed"]
        t = rec["torque"]
        robot.set_speed([s, s, s, s], overwrite_speeds=True)
        robot.set_torque_limit([t, t, t, t])
        robot.move_j(rec["j1"], rec["j2"], rec["j3"], rec["j4"])
        time.sleep(dwell_after_move_s)
        if robot.gripper_open_flag != rec["gripper_open"]:
            robot.actuate_gripper()
            time.sleep(dwell_after_grip_s)


def play_saved_history(robot, history_path=TEACH_HISTORY_PATH, limits_path=TEACH_LIMITS_PATH):
    """Enable torque without jumping, then play poses saved by teach mode."""
    if not os.path.isfile(history_path):
        print(f"No saved poses at {history_path}. Record some with: python MyRobot.py")
        return
    if os.path.isfile(limits_path):
        load_joint_limits(robot, limits_path)
    load_movement_history(robot, history_path)
    for rec in robot.movement_history:
        pose = [rec["j1"], rec["j2"], rec["j3"], rec["j4"]]
        for i in range(4):
            robot.joint_limits[i, 0] = min(robot.joint_limits[i, 0], pose[i])
            robot.joint_limits[i, 1] = max(robot.joint_limits[i, 1], pose[i])
    robot.hold_present_position()
    robot.enable_motors()
    play_configuration_history(robot)


def teach(robot, poll_period=0.15, limits_path=TEACH_LIMITS_PATH, history_path=TEACH_HISTORY_PATH):
    """Torque stays off. Move the arm by hand.

    The terminal shows the live joint angles and the min/max seen so far.
    r  record this pose for playback
    u  delete the last recorded pose
    c  restart min/max from the current pose
    q  save and quit
    """
    if os.path.isfile(history_path):
        try:
            load_movement_history(robot, history_path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Could not load existing history: {e}")

    joint_min = None
    joint_max = None
    if os.path.isfile(limits_path):
        try:
            with open(limits_path, encoding="utf-8") as f:
                previous = json.load(f)
            joint_min = np.array(previous["min_deg"], dtype=float)
            joint_max = np.array(previous["max_deg"], dtype=float)
            print("Continuing joint extremes from the last session. Press c to start over.")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not load existing limits: {e}")

    print(
        "\nMove each joint to its mechanical stop and back, watching min/max.\n"
        "Commands: [r] record pose   [u] undo last pose   [c] reset min/max   [q] save and quit\n"
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            try:
                angles = robot.read_joint_angles()
            except Exception as e:
                sys.stdout.write(f"\rRead error: {e}{' ' * 20}")
                sys.stdout.flush()
                time.sleep(poll_period)
                angles = None

            if angles is not None:
                if joint_min is None:
                    joint_min = angles.copy()
                    joint_max = angles.copy()
                else:
                    joint_min = np.minimum(joint_min, angles)
                    joint_max = np.maximum(joint_max, angles)
                now = " ".join(f"{v:7.2f}" for v in angles)
                lo = " ".join(f"{v:7.2f}" for v in joint_min)
                hi = " ".join(f"{v:7.2f}" for v in joint_max)
                sys.stdout.write(
                    f"\rnow [{now}]  min [{lo}]  max [{hi}]  poses {len(robot.movement_history):3d}  "
                )
                sys.stdout.flush()

            ready, _, _ = select.select([sys.stdin], [], [], poll_period)
            if not ready:
                continue
            key = sys.stdin.read(1).lower()
            if key == "r" and angles is not None:
                sys.stdout.write("\n")
                robot.joint_angles = angles.copy()
                record_configuration(robot)
                save_movement_history(robot, history_path)
            elif key == "u":
                sys.stdout.write("\n")
                delete_last_recorded_configuration(robot)
                save_movement_history(robot, history_path)
            elif key == "c" and angles is not None:
                joint_min = angles.copy()
                joint_max = angles.copy()
                sys.stdout.write("\nMin/max reset to the current pose.\n")
            elif key == "q":
                sys.stdout.write("\n")
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if joint_min is not None:
        save_joint_limits(robot, joint_min, joint_max, limits_path)
        print("Furthest angles reached (degrees):")
        for i in range(4):
            print(f"  J{i + 1}: min {joint_min[i]:7.2f}   max {joint_max[i]:7.2f}")
        print(
            "\nThese are only the extremes you actually moved to. "
            "Sweep every joint to the physical stop before trusting them.\n"
            "Stay a few degrees inside this range when you command the arm.\n"
            "Play the recorded poses later with:  python MyRobot.py --play"
        )
    save_movement_history(robot, history_path)
