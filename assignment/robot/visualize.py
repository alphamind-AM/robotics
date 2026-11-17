"""Top and arm-plane views of the commanded arm. Does not open the robot.

The stick is the origin of each frame from the kinematics, so the picture
uses the same lengths and joint zeros as inverse() and forward().
"""

import numpy as np

from robot.kinematics import (
    JOINT_ZERO_DEG,
    _arm_frames,
    cosd,
    forward,
    inverse,
    sind,
)


TOOL_LENGTH = 30.0


def link_points(joint_angles_deg):
    """XYZ of the base and the four joint frames, in millimetres."""
    frames = _arm_frames(joint_angles_deg)
    return np.array([frame[:3, 3] for frame in frames], dtype=float)


def tool_segment(joint_angles_deg, length=TOOL_LENGTH):
    """Short line from the tip along the tool x axis."""
    pose = _arm_frames(joint_angles_deg)[-1]
    tip = pose[:3, 3]
    end = tip + float(length) * pose[:3, 0]
    return np.vstack([tip, end])


def arm_plane_axis(tip_xyz, yaw_deg):
    """Unit vector in xy for the side view.

    The tip points to the positive side, so the side view matches the top
    view. Joint 1 at ±180° still aims the model at +x or -x, and using that
    yaw as the axis draws the stick on the opposite side of the top view.
    Straight up, the tip has no horizontal direction, so the joint yaw is used.
    """
    tip = np.asarray(tip_xyz, dtype=float).reshape(3)
    horizontal = float(np.hypot(tip[0], tip[1]))
    if horizontal < 1.0:
        return cosd(yaw_deg), sind(yaw_deg)
    return tip[0] / horizontal, tip[1] / horizontal


def project_reach(points_xyz, axis):
    """Horizontal coordinate along one arm-plane axis. Negative is behind the base."""
    ux, uy = axis
    points = np.asarray(points_xyz, dtype=float)
    return points[..., 0] * ux + points[..., 1] * uy


def _yaw_deg(joint_angles_deg):
    joints = np.asarray(joint_angles_deg, dtype=float).reshape(4)
    return float(joints[0] + JOINT_ZERO_DEG[0])


class _RecordingRobot:
    """Stores the joint command each mission step would send."""

    def __init__(self):
        self.label = "step"
        self.frames = []

    def move_c(self, x, y, z, pitch_deg, **kwargs):
        joints = np.asarray(inverse(x, y, z, pitch_deg), dtype=float).reshape(4)
        self.frames.append((joints, self.label))

    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, **kwargs):
        joints = np.asarray([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float)
        self.frames.append((joints, self.label))

    def execute(self, samples):
        for sample in samples:
            if isinstance(sample, (tuple, list)) and len(sample) == 2 and np.isscalar(sample[0]):
                joints = sample[1]
            else:
                joints = sample
            joints = np.asarray(joints, dtype=float).reshape(4)
            self.frames.append((joints, self.label))


def collect_mission(mission, context=None):
    """Run a mission against a recorder. Returns (frames, error).

    context is shared with the steps, so a pose chosen at run time is
    what gets drawn.
    """
    robot = _RecordingRobot()

    def announce(index, step):
        robot.label = getattr(step, "label", "step")

    error = None
    try:
        mission.run(robot, context=context, announce=announce)
    except (ValueError, NotImplementedError) as exc:
        error = str(exc)
    return robot.frames, error


def _pose_line(joints):
    pose = forward(joints)
    return (
        f"x={pose[0]:7.1f}   y={pose[1]:7.1f}   "
        f"z={pose[2]:7.1f}   pitch={pose[3]:6.1f}°"
    )


def _joint_line(joints):
    joints = np.asarray(joints, dtype=float).reshape(4)
    return "   ".join(f"j{i + 1}={angle:7.1f}°" for i, angle in enumerate(joints))


def _limits(values, pad_fraction, minimum_span):
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < minimum_span:
        mid = 0.5 * (low + high)
        low = mid - 0.5 * minimum_span
        high = mid + 0.5 * minimum_span
    pad = pad_fraction * (high - low)
    return low - pad, high + pad


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for --view.\n"
            "From the repository root, with the virtual environment active:\n"
            "  pip install matplotlib"
        ) from exc
    return plt


def show_frames(frames, title, notice=None):
    """Open the two views. frames is a list of (joints, label).

    Left and right step through commands. Space plays them in order.
    The window starts on the first command.
    """
    if len(frames) == 0:
        print(title)
        if notice:
            print(notice)
        print("Nothing to draw.")
        return

    plt = _pyplot()
    tips = []
    reaches = []
    heights = []
    for joints, _label in frames:
        tip = link_points(joints)[-1]
        tips.append(tip)
        reaches.append(float(project_reach(tip, arm_plane_axis(tip, _yaw_deg(joints)))))
        heights.append(float(tip[2]))
    tips = np.asarray(tips, dtype=float)
    reaches = np.asarray(reaches, dtype=float)
    heights = np.asarray(heights, dtype=float)

    sticks = [link_points(joints) for joints, _label in frames]
    tools = [tool_segment(joints) for joints, _label in frames]
    all_xyz = np.vstack(sticks + tools + [tips])
    reach_parts = []
    for stick, tool, (joints, _label) in zip(sticks, tools, frames):
        axis = arm_plane_axis(stick[-1], _yaw_deg(joints))
        reach_parts.append(project_reach(stick, axis))
        reach_parts.append(project_reach(tool, axis))
    reach_parts.append(reaches)
    reach_values = np.concatenate(reach_parts)
    z_values = np.concatenate(
        [stick[:, 2] for stick in sticks] + [tool[:, 2] for tool in tools] + [heights]
    )

    x_lim = _limits(all_xyz[:, 0], 0.08, 160.0)
    y_lim = _limits(all_xyz[:, 1], 0.08, 160.0)
    r_lim = _limits(reach_values, 0.08, 160.0)
    z_lim = _limits(np.append(z_values, 0.0), 0.08, 160.0)

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(11.0, 6.2))
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    status = fig.text(0.5, 0.06, "", ha="center", va="center", family="monospace", fontsize=11)
    fig.text(
        0.5,
        0.015,
        "left / right: step     space: play or pause     q: close",
        ha="center",
        va="center",
        fontsize=9,
        color="0.35",
    )

    state = {"index": 0, "playing": False}

    def draw():
        index = state["index"]
        joints, label = frames[index]
        stick = sticks[index]
        tool = tools[index]
        axis = arm_plane_axis(stick[-1], _yaw_deg(joints))
        reach = project_reach(stick, axis)
        tool_reach = project_reach(tool, axis)

        ax_top.clear()
        ax_side.clear()

        if index > 0:
            ax_top.plot(tips[: index + 1, 0], tips[: index + 1, 1], color="C1", lw=1.5)
            ax_side.plot(reaches[: index + 1], heights[: index + 1], color="C1", lw=1.5)
        if index + 1 < len(frames):
            ax_top.plot(tips[index:, 0], tips[index:, 1], color="0.75", lw=1.0)
            ax_side.plot(reaches[index:], heights[index:], color="0.75", lw=1.0)

        ax_top.plot(stick[:, 0], stick[:, 1], "-o", color="C0", ms=5, lw=2.0)
        ax_top.plot(tool[:, 0], tool[:, 1], color="C2", lw=2.0)
        ax_top.plot(0.0, 0.0, "+", color="0.2", ms=10)
        ax_side.plot(reach, stick[:, 2], "-o", color="C0", ms=5, lw=2.0)
        ax_side.plot(tool_reach, tool[:, 2], color="C2", lw=2.0)
        ax_side.plot(0.0, 0.0, "+", color="0.2", ms=10)

        ax_top.set_title("Top view")
        ax_top.set_xlabel("x (mm)")
        ax_top.set_ylabel("y (mm)")
        ax_top.set_xlim(*x_lim)
        ax_top.set_ylim(*y_lim)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.grid(True, alpha=0.4)
        ax_top.axhline(0.0, color="0.85", lw=0.8)
        ax_top.axvline(0.0, color="0.85", lw=0.8)

        ax_side.set_title("Arm plane")
        ax_side.set_xlabel("reach (mm)")
        ax_side.set_ylabel("z (mm)")
        ax_side.set_xlim(*r_lim)
        ax_side.set_ylim(*z_lim)
        ax_side.set_aspect("equal", adjustable="box")
        ax_side.grid(True, alpha=0.4)
        ax_side.axhline(0.0, color="0.85", lw=0.8)
        ax_side.axvline(0.0, color="0.85", lw=0.8)

        play = "playing" if state["playing"] else "paused"
        fig.suptitle(f"{title}    {index + 1}/{len(frames)}  {label}  ({play})")
        lines = _joint_line(joints) + "\n" + _pose_line(joints)
        if notice:
            lines += "\n" + notice
        status.set_text(lines)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n"):
            state["playing"] = False
            state["index"] = min(state["index"] + 1, len(frames) - 1)
            draw()
        elif event.key in ("left", "p"):
            state["playing"] = False
            state["index"] = max(state["index"] - 1, 0)
            draw()
        elif event.key == " ":
            state["playing"] = not state["playing"]
            draw()
        elif event.key == "q":
            plt.close(fig)

    def tick():
        if not state["playing"] or len(frames) < 2:
            return
        state["index"] = (state["index"] + 1) % len(frames)
        if state["index"] == 0:
            state["playing"] = False
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(tick)
    timer.start()
    draw()
    plt.show()


def view_mission(mission, context=None):
    """Draw every joint command in a mission. Does not connect to the robot."""
    print(f"Viewing {mission.name}. The robot is not connected.")
    frames, error = collect_mission(mission, context)
    notice = None
    if error:
        notice = f"stopped: {error}"
        print(notice)
    print(f"{len(frames)} commands")
    show_frames(frames, mission.name, notice)


def view_configuration(joint_angles_deg=None, pose=None):
    """Draw one commanded pose. Joints win when both are given."""
    print("Arm view. The robot is not connected.")
    notice = None
    if joint_angles_deg is not None:
        joints = np.asarray(joint_angles_deg, dtype=float).reshape(4)
        label = "joints"
        if pose is not None:
            notice = "showing the given joint angles"
    elif pose is not None:
        target = np.asarray(pose, dtype=float).reshape(4)
        try:
            joints = np.asarray(inverse(*target), dtype=float).reshape(4)
        except (ValueError, NotImplementedError) as exc:
            print(f"stopped: {exc}")
            return
        label = "pose"
    else:
        joints = np.zeros(4)
        label = "zeros"
        notice = "no angles given, showing joints 0 0 0 0"
    print(_joint_line(joints))
    print(_pose_line(joints))
    show_frames([(joints, label)], "arm", notice)
