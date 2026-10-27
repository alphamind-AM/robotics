import argparse

from robot.arm import MyRobot


def main():
    parser = argparse.ArgumentParser(description="Hand-teach the arm, or play back saved poses.")
    parser.add_argument("--play", action="store_true", help="Play teach_history.json")
    parser.add_argument("--vision", action="store_true", help="Laptop camera only, for vision development")
    parser.add_argument("--camera", default=None, help="Video device, e.g. /dev/video0 or /dev/video4")
    parser.add_argument("--no-camera", action="store_true", help="Do not open a camera window")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print joint angles and end-effector position. Does not connect to the robot.",
    )
    parser.add_argument(
        "--joints",
        nargs=4,
        type=float,
        metavar=("J1", "J2", "J3", "J4"),
        help="Joint angles in degrees for forward kinematics",
    )
    parser.add_argument(
        "--pose",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "PITCH"),
        help="Target x y z pitch_deg for inverse kinematics",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Camera depth, shifted by the end-effector camera offset",
    )
    parser.add_argument(
        "--mission",
        default=None,
        help="Mission to run, e.g. square_with_x. With --check, only print the path.",
    )
    args = parser.parse_args()

    if args.mission:
        from missions import MISSIONS
        from robot.mission import preview_mission

        if args.mission not in MISSIONS:
            known = ", ".join(sorted(MISSIONS))
            print(f"Unknown mission {args.mission}. Known missions: {known}")
            return
        mission = MISSIONS[args.mission]
        if args.check:
            preview_mission(mission)
            return
        robot = MyRobot(home_on_start=False, torque_on_start=False)
        if not robot.init_status:
            return
        try:
            # Lock the pose the arm is already in, then turn torque on.
            # A goal written while torque is off is where the arm jumps.
            robot.set_speed(robot.motor_speed, overwrite_speeds=True)
            robot.set_torque_limit(robot.motor_torque)
            robot.hold_present_position()
            robot.enable_motors()
            mission.run(robot)
        finally:
            if robot.init_status:
                robot.disable_motors()
        return

    if args.check or args.joints or args.pose or args.depth is not None:
        from robot.kinematics import preview
        preview(args.joints, args.pose, args.depth)
        return

    if args.vision:
        from robot.camera import Camera
        from robot.perception import detect_faces
        camera = Camera()
        try:
            camera.preview_detection(detect_faces, device=args.camera)
        finally:
            camera.stop()
        return

    robot = MyRobot(home_on_start=False, torque_on_start=False)
    if not robot.init_status:
        return
    try:
        if not args.no_camera:
            robot.start_camera(args.camera)
        if args.play:
            robot.play_saved_history()
        else:
            robot.teach()
    finally:
        robot.stop_camera()
        if robot.init_status:
            robot.disable_motors()


if __name__ == "__main__":
    main()
