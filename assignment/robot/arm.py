import time

import numpy as np

from robot.bus import DynamixelBus
from robot.camera import Camera
from robot.kinematics import forward, inverse, signed_deg


class MyRobot:
    """Joint-level arm. Asks the bus for bytes and kinematics for poses."""

    def __init__(self, home_on_start=True, torque_on_start=True, device_name=None):
        self.motor_ids = [1, 2, 3, 4]
        self.gripper_motor_id = 5
        self.forward_transform = np.zeros((4, 4))
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
        self.joint_pos = np.zeros((4, 4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True

        self.joint_limits = np.array([[-130, 130], [-180, 180], [-140, 140], [-140, 140]])
        self.ik_weights = np.array([0.25, 0.25, 0.25, 1, 1, 1])
        self.joint_offsets = np.array([150, 150, 150, 150], dtype=float)
        self.joint_angle_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.init_status = 0

        self.movement_history = []
        self.motor_speed = np.array([0.1, 0.1, 0.1, 0.1])
        self.motor_torque = np.array([1.0, 1.0, 1.0, 1.0])
        self.pitch = 0.0

        self.bus = DynamixelBus(device_name)
        self.camera = Camera()

        try:
            self.bus.open()
            # Torque-off has to be the first bus command in teach mode.
            # A goal written while torque is off is where the arm jumps
            # when torque is turned back on.
            if torque_on_start:
                self.set_speed(self.motor_speed, overwrite_speeds=True)
                self.set_torque_limit(self.motor_torque)
                self.hold_present_position()
                self.enable_motors()
                if home_on_start:
                    self.move_j(0, 0, 0, 0)
            else:
                if not self.release_motors():
                    raise RuntimeError("Could not turn torque off")
            self.init_status = 1
        except Exception as e:
            print(f"Initialization error: {e}")
            self.init_status = 0

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _check_limits_one(self, deg, idx):
        lo, hi = self.joint_limits[idx]
        if not (lo <= deg <= hi):
            if idx == 0:
                raise ValueError(f"Angle Limits for first Axis Reached, Min/Max: [{lo}°, {hi}°]" + f" (received {deg}°)")
            elif idx == 1:
                raise ValueError(f"Angle Limits for second Axis Reached, Min/Max: [{lo}°, {hi}°]" + f" (received {deg}°)"   )
            else:
                raise ValueError(f"Angle Limits Reached, Min/Max: [{lo}°, {hi}°]")
        return deg

    def set_speed(self, speeds, overwrite_speeds=True):
        speeds = np.asarray(speeds, dtype=float)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            s = float(speeds[i])
            if 0 < s <= 1:
                raw = int(round(s * 1023))
                self.bus.write2(dxl_id, self.bus.ADDR_MOVING_SPEED, raw)
            else:
                print("Movement speed out of range, enter value between (0, 1].")

    def set_torque_limit(self, torques):
        torques = np.asarray(torques, dtype=float)
        self.motor_torque = torques.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            t = float(torques[i])
            if 0 < t <= 1:
                self.bus.write2(dxl_id, self.bus.ADDR_TORQUE_LIMIT, int(round(t * 1023)))
            else:
                print("Torque limit out of range, enter value between (0, 1].")

    def smooth_speed(self, joint_angles_delta_deg):
        max_angle = float(np.max(np.abs(joint_angles_delta_deg)))
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg != 0:
            new_speeds = np.abs(joint_angles_delta_deg / speed_per_deg) * 0.01
            for i in range(len(self.motor_speed)):
                if new_speeds[i] == 0:
                    new_speeds[i] = self.motor_speed[i]
                else:
                    new_speeds[i] = new_speeds[i] * self.motor_speed[i]
            self.set_speed(new_speeds, overwrite_speeds=False)

    def enable_motors(self):
        for dxl_id in self.motor_ids:
            self.bus.write1(dxl_id, self.bus.ADDR_TORQUE_ENABLE, self.bus.TORQUE_ENABLE)
        print("Dynamixel torque enabled on all motors.")

    def release_motors(self, announce=True):
        """Turn torque off but keep the serial port open so positions can still be read."""
        ok = True
        for dxl_id in self.motor_ids:
            try:
                self.bus.write1(dxl_id, self.bus.ADDR_TORQUE_ENABLE, self.bus.TORQUE_DISABLE)
            except Exception as e:
                ok = False
                print(f"Release motor {dxl_id} error: {e}")
        if announce and ok:
            print("Torque off. You can move the arm by hand.")
        return ok

    def disable_motors(self):
        self.release_motors(announce=False)
        try:
            self.bus.close()
        finally:
            self.init_status = 0

    def get_position(self, motor_id):
        ticks = self.bus.read2(motor_id, self.bus.ADDR_PRESENT_POSITION)
        return self.bus.ticks_to_deg(ticks)

    def read_joint_angles(self):
        j_a = np.zeros(4, dtype=float)
        for i, dxl_id in enumerate(self.motor_ids):
            ticks = self.bus.read2(dxl_id, self.bus.ADDR_PRESENT_POSITION)
            deg = self.bus.ticks_to_deg(ticks) - self.joint_offsets[i]
            j_a[i] = deg
            self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a

    def hold_present_position(self):
        """Write the current pose as the goal so enabling torque does not jump the arm."""
        ja = self.read_joint_angles()
        self.joint_angles = ja.copy()
        cmd_deg = ja + self.joint_offsets
        for i, dxl_id in enumerate(self.motor_ids):
            self.bus.write2(dxl_id, self.bus.ADDR_GOAL_POSITION, self.bus.deg_to_ticks(cmd_deg[i]))
        return ja

    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, wait=True, tol_deg=2.0, poll_period=0.05):
        desired = signed_deg(np.array([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float))
        for i in range(4):
            desired[i] = self._check_limits_one(desired[i], i)

        if self.use_smooth_speed_flag:
            self.smooth_speed(desired - self.joint_angles)

        cmd_deg = desired + self.joint_offsets
        for i, dxl_id in enumerate(self.motor_ids):
            self.bus.write2(dxl_id, self.bus.ADDR_GOAL_POSITION, self.bus.deg_to_ticks(cmd_deg[i]))

        if not wait:
            return

        while True:
            try:
                ja = self.read_joint_angles()
            except Exception:
                time.sleep(poll_period)
                continue
            if np.all(np.abs(ja - desired) < tol_deg):
                break
            time.sleep(poll_period)

    def execute(self, samples):
        """Follow samples from robot.trajectory. Each sample is joint angles or (time_s, joints)."""
        for sample in samples:
            joints = _joints_of(sample)
            self.move_j(joints[0], joints[1], joints[2], joints[3])

    def move_c(self, x, y, z, pitch_deg, **kwargs):
        j_deg = inverse(x, y, z, pitch_deg)
        self.move_j(j_deg[0], j_deg[1], j_deg[2], j_deg[3], **kwargs)

    def read_ee_position(self):
        return forward(self.read_joint_angles())

    def record_configuration(self):
        from robot import teach
        return teach.record_configuration(self)

    def delete_last_recorded_configuration(self):
        from robot import teach
        return teach.delete_last_recorded_configuration(self)

    def record_measured_configuration(self):
        from robot import teach
        return teach.record_measured_configuration(self)

    def save_movement_history(self, path=None):
        from robot import teach
        if path is None:
            path = teach.TEACH_HISTORY_PATH
        return teach.save_movement_history(self, path)

    def load_movement_history(self, path=None):
        from robot import teach
        if path is None:
            path = teach.TEACH_HISTORY_PATH
        return teach.load_movement_history(self, path)

    def save_joint_limits(self, joint_min, joint_max, path=None):
        from robot import teach
        if path is None:
            path = teach.TEACH_LIMITS_PATH
        return teach.save_joint_limits(self, joint_min, joint_max, path)

    def load_joint_limits(self, path=None):
        from robot import teach
        if path is None:
            path = teach.TEACH_LIMITS_PATH
        return teach.load_joint_limits(self, path)

    def play_configuration_history(self, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
        from robot import teach
        return teach.play_configuration_history(self, dwell_after_move_s, dwell_after_grip_s)

    def play_saved_history(self, history_path=None, limits_path=None):
        from robot import teach
        if history_path is None:
            history_path = teach.TEACH_HISTORY_PATH
        if limits_path is None:
            limits_path = teach.TEACH_LIMITS_PATH
        return teach.play_saved_history(self, history_path, limits_path)

    def teach(self, poll_period=0.15, limits_path=None, history_path=None):
        from robot import teach as teach_mod
        if limits_path is None:
            limits_path = teach_mod.TEACH_LIMITS_PATH
        if history_path is None:
            history_path = teach_mod.TEACH_HISTORY_PATH
        return teach_mod.teach(self, poll_period, limits_path, history_path)

    def find_camera_device(self):
        from robot.camera import find_camera_device
        return find_camera_device()

    def start_camera(self, device=None):
        return self.camera.start(device)

    def stop_camera(self):
        self.camera.stop()


def _joints_of(sample):
    if isinstance(sample, (tuple, list)) and len(sample) == 2 and np.isscalar(sample[0]):
        joints = sample[1]
    else:
        joints = sample
    return np.asarray(joints, dtype=float).reshape(4)
