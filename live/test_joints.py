"""
test_joints.py -- Diagnostic: move each joint one at a time to identify motor mapping.

Sends a small movement to each joint individually while holding all others
at the stow position. Watch which physical motor moves and report back.

Usage (from project root):
    python -m live.test_joints --port /dev/tty.usbmodem5AB90657441
    python -m live.test_joints --port /dev/tty.usbmodem5AB90652661 --arm left
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Path setup
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _live = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_root)
    for p in [_live]:
        if p not in sys.path:
            sys.path.insert(0, p)

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

_PROJECT_ROOT = Path(__file__).parent.parent
_CALIBRATION_DIR = _PROJECT_ROOT / "robo-ops" / "calibration" / "follower"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Neutral position (all joints at 0 degrees except gripper at 50%)
NEUTRAL = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

# How far to move each joint for the test (degrees, except gripper which is %)
TEST_AMPLITUDE = {
    "shoulder_pan": 30.0,
    "shoulder_lift": 30.0,
    "elbow_flex": 30.0,
    "wrist_flex": 30.0,
    "wrist_roll": 30.0,
    "gripper": 40.0,
}


def move_smooth(robot, target, steps=40, fps=30):
    """Smoothly interpolate to target position."""
    obs = robot.get_observation()
    current = {f"{j}.pos": obs.get(f"{j}.pos", 0.0) for j in JOINTS}
    dt = 1.0 / fps

    for step in range(1, steps + 1):
        alpha = step / steps
        waypoint = {}
        for key in target:
            waypoint[key] = current[key] + alpha * (target[key] - current[key])
        robot.send_action(waypoint)
        time.sleep(dt)


def main():
    p = argparse.ArgumentParser(description="Test each joint individually")
    p.add_argument("--port", type=str, required=True, help="Serial port")
    p.add_argument("--arm", type=str, default="right", choices=["right", "left"])
    args = p.parse_args()

    cal_id = f"bimanual_follower_{args.arm}"
    config = SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        id=cal_id,
        calibration_dir=_CALIBRATION_DIR,
    )
    robot = SO101Follower(config)

    print(f"Connecting {args.arm} arm on {args.port}...")
    robot.connect()

    # Read current positions
    obs = robot.get_observation()
    print("\nCurrent positions (degrees):")
    for j in JOINTS:
        print(f"  {j:20s}: {obs.get(f'{j}.pos', 0.0):+8.1f}")

    # Move to neutral first
    print("\nMoving to neutral position (all joints to 0)...")
    neutral_action = {f"{j}.pos": v for j, v in NEUTRAL.items()}
    move_smooth(robot, neutral_action, steps=60)
    print("At neutral.\n")
    time.sleep(0.5)

    # Test each joint
    for joint in JOINTS:
        amp = TEST_AMPLITUDE[joint]
        input(f"Press Enter to test [{joint}] (will move +{amp} then back)... ")

        # Build target: neutral + this joint offset
        target_pos = {f"{j}.pos": NEUTRAL[j] for j in JOINTS}
        target_pos[f"{joint}.pos"] = NEUTRAL[joint] + amp

        print(f"  Moving {joint} to +{amp}...")
        move_smooth(robot, target_pos, steps=30)
        time.sleep(0.5)

        # Return to neutral
        print(f"  Returning {joint} to neutral...")
        move_smooth(robot, neutral_action, steps=30)
        time.sleep(0.3)

        actual = input(f"  Which physical motor moved? (type what you saw): ")
        print(f"  >> {joint} (Motor ID {JOINTS.index(joint) + 1}) -> {actual}\n")

    # Stow before disconnect
    print("Stowing...")
    stow_target = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -99.0,
        "elbow_flex.pos": 99.0,
        "wrist_flex.pos": -99.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 80.0,
    }
    move_smooth(robot, stow_target, steps=50)

    print("Disconnecting...")
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
