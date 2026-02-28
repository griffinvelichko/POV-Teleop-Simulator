"""
main.py -- Live POV teleop with real SO-ARM101 robot(s) via LeRobot.

Pipeline: Camera -> Pose -> Mapping -> Smoother -> RobotController(s) -> LiveDisplay

Usage (from project root):
    python -m live.main --dry-run                                    # no robot, prints actions
    python -m live.main --right-port /dev/ttyACM0 --right-only      # single right arm
    python -m live.main --right-port /dev/ttyACM0 --left-port /dev/ttyACM1  # dual arm
    python -m live.main --alpha 0.3 --max-step 3.0                  # tune smoothing / safety
"""

import argparse
import os
import signal
import sys
import time

# Path setup: add src/ and live/ so all imports resolve
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_root, "src")
    _live = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_root)
    for p in [_src, _live]:
        if p not in sys.path:
            sys.path.insert(0, p)

import numpy as np

# From src/ (reused as-is)
from camera import Camera
from config import CAMERA_DEVICE, HOME_POSITION, SMOOTHING_ALPHA
from mapping import JointMapper, LeftJointMapper
from pose import PoseTracker, draw_landmarks_on_frame
from smoother import Smoother

# From live/
from display import LiveDisplay
from robot import RobotController

# Defaults
DEFAULT_RIGHT_PORT = "/dev/tty.usbmodem5AB90657441"
DEFAULT_LEFT_PORT = "/dev/tty.usbmodem5AB90652661"
DEFAULT_MAX_STEP = 5.0
HOMING_DURATION_S = 2.0
POSE_TIMEOUT_S = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="POV Teleop: Live robot control via LeRobot")
    p.add_argument("--camera", type=int, default=CAMERA_DEVICE, help="Camera device index")
    p.add_argument("--right-port", type=str, default=DEFAULT_RIGHT_PORT, help="Serial port for right arm")
    p.add_argument("--left-port", type=str, default=DEFAULT_LEFT_PORT, help="Serial port for left arm")
    p.add_argument("--alpha", type=float, default=SMOOTHING_ALPHA, help="Smoother EMA alpha")
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP, help="Max degrees per step (safety)")
    p.add_argument("--no-home", action="store_true", help="Skip homing on startup")
    p.add_argument("--dry-run", action="store_true", help="Run pipeline without robot (print actions)")
    p.add_argument("--right-only", action="store_true", help="Use right arm only")
    p.add_argument("--left-only", action="store_true", help="Use left arm only")
    p.add_argument("--no-stow", action="store_true", help="Skip stow on exit (debug)")
    return p.parse_args()


def main():
    args = parse_args()

    use_right = not args.left_only
    use_left = not args.right_only

    # -- Initialize shared components --
    print("Initializing camera...")
    camera = Camera(device=args.camera)

    print("Initializing pose tracker...")
    tracker = PoseTracker()

    print("Initializing mappers and smoothers...")
    right_mapper = JointMapper()
    left_mapper = LeftJointMapper()
    right_smoother = Smoother(alpha=args.alpha)
    left_smoother = Smoother(alpha=args.alpha)

    # -- Initialize robot controllers --
    right_robot: RobotController | None = None
    left_robot: RobotController | None = None

    if not args.dry_run:
        if use_right:
            print(f"Connecting right arm on {args.right_port}...")
            right_robot = RobotController(
                port=args.right_port,
                max_relative_target=args.max_step,
                arm="right",
            )
            right_robot.connect()

        if use_left:
            print(f"Connecting left arm on {args.left_port}...")
            left_robot = RobotController(
                port=args.left_port,
                max_relative_target=args.max_step,
                arm="left",
            )
            left_robot.connect()

        # Homing
        if not args.no_home:
            if right_robot is not None:
                right_robot.home(HOME_POSITION, duration_s=HOMING_DURATION_S)
            if left_robot is not None:
                left_robot.home(HOME_POSITION, duration_s=HOMING_DURATION_S)
    else:
        print("[DRY RUN] No robot connection. Actions printed to stdout.")

    display = LiveDisplay()

    # -- State --
    home = np.array(HOME_POSITION, dtype=np.float64)
    last_right_action = home.copy()
    last_left_action = home.copy()
    last_right_pose_time = time.time()
    last_left_pose_time = time.time()
    frozen = False
    frame_count = 0
    t_start = time.time()
    fps = 0.0

    # -- Graceful shutdown --
    shutdown_requested = False

    def _shutdown_handler(signum, _frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\n[signal] Shutdown requested...")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    arms_str = []
    if use_right:
        arms_str.append("right")
    if use_left:
        arms_str.append("left")
    print(f"\nPipeline ready ({', '.join(arms_str)} arm{'s' if len(arms_str) > 1 else ''}).")
    print("Move your arms in front of the camera.")
    print("Keys: q=quit  e=e-stop  h=re-home  f=freeze  s=stow\n")

    try:
        while not shutdown_requested:
            ok, frame = camera.read()
            if not ok:
                print("Camera read failed.")
                break

            timestamp_ms = int((time.time() - t_start) * 1000)
            pose_result = tracker.process(frame, timestamp_ms)
            now = time.time()

            # -- Right arm --
            right_action = last_right_action
            if use_right:
                raw_right = right_mapper.compute(pose_result)
                if raw_right is not None:
                    right_action = right_smoother.update(raw_right)
                    last_right_action = right_action
                    last_right_pose_time = now
                else:
                    right_action = last_right_action

            # -- Left arm --
            left_action = last_left_action
            if use_left:
                raw_left = left_mapper.compute(pose_result)
                if raw_left is not None:
                    left_action = left_smoother.update(raw_left)
                    last_left_action = left_action
                    last_left_pose_time = now
                else:
                    left_action = last_left_action

            # -- Watchdog: freeze if no pose for too long --
            right_stale = (now - last_right_pose_time) > POSE_TIMEOUT_S
            left_stale = (now - last_left_pose_time) > POSE_TIMEOUT_S
            if use_right and use_left:
                watchdog_triggered = right_stale and left_stale
            elif use_right:
                watchdog_triggered = right_stale
            else:
                watchdog_triggered = left_stale

            if watchdog_triggered and not frozen:
                print("[watchdog] No pose detected, freezing.")
                frozen = True

            # -- Send to robots --
            if not args.dry_run and not frozen:
                if right_robot is not None:
                    right_robot.step(right_action)
                if left_robot is not None:
                    left_robot.step(left_action)
            elif not args.dry_run and frozen:
                if right_robot is not None:
                    right_robot.freeze()
                if left_robot is not None:
                    left_robot.freeze()

            if args.dry_run:
                r_deg = np.degrees(right_action)
                l_deg = np.degrees(left_action)
                print(
                    f"R: {' '.join(f'{d:+6.1f}' for d in r_deg)}  "
                    f"L: {' '.join(f'{d:+6.1f}' for d in l_deg)}"
                )

            # -- Read actual positions for display --
            right_actual = None
            left_actual = None
            if not args.dry_run:
                try:
                    if right_robot is not None:
                        right_actual = right_robot.get_joint_positions()
                except Exception:
                    pass
                try:
                    if left_robot is not None:
                        left_actual = left_robot.get_joint_positions()
                except Exception:
                    pass

            # -- Draw skeleton --
            draw_landmarks_on_frame(frame, pose_result)

            # -- FPS --
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed > 0:
                fps = frame_count / elapsed

            # -- Display --
            display_frame = display.render(
                camera_frame=frame,
                right_cmd=right_action if use_right else None,
                right_actual=right_actual,
                left_cmd=left_action if use_left else None,
                left_actual=left_actual,
                fps=fps,
                right_connected=(right_robot is not None and right_robot.connected),
                left_connected=(left_robot is not None and left_robot.connected),
                frozen=frozen,
            )
            key = display.show(display_frame)

            # -- Key handling --
            if key == ord("q") or key == 27:  # quit / ESC
                break
            elif key == ord("e"):  # emergency stop
                print("[E-STOP] Disabling torque on all arms!")
                if right_robot is not None:
                    right_robot.disconnect()
                if left_robot is not None:
                    left_robot.disconnect()
                break
            elif key == ord("h"):  # re-home
                if not frozen:
                    print("[home] Re-homing...")
                    if right_robot is not None and right_robot.connected:
                        right_robot.home(HOME_POSITION, duration_s=HOMING_DURATION_S)
                    if left_robot is not None and left_robot.connected:
                        left_robot.home(HOME_POSITION, duration_s=HOMING_DURATION_S)
            elif key == ord("f"):  # toggle freeze
                frozen = not frozen
                print(f"[freeze] {'FROZEN' if frozen else 'UNFROZEN'}")
                if not frozen:
                    last_right_pose_time = now
                    last_left_pose_time = now
            elif key == ord("s"):  # manual stow (keeps running after)
                print("[stow] Stowing arms...")
                frozen = True
                if right_robot is not None and right_robot.connected:
                    right_robot.stow()
                if left_robot is not None and left_robot.connected:
                    left_robot.stow()

    except Exception as e:
        print(f"\n[error] {e}")
        import traceback

        traceback.print_exc()

    # -- Cleanup --
    print("\nShutting down...")
    camera.release()
    tracker.close()
    if right_robot is not None and right_robot.connected:
        if args.no_stow:
            right_robot.disconnect()
        else:
            right_robot.stow_and_disconnect()
    if left_robot is not None and left_robot.connected:
        if args.no_stow:
            left_robot.disconnect()
        else:
            left_robot.stow_and_disconnect()
    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
