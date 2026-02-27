"""
calibrate.py -- Calibration helper for SO-ARM101 via LeRobot.

Wraps lerobot-calibrate with project-specific instructions.
Run once per arm before first use, or after reassembling.

Usage (from project root):
    python -m live.calibrate --port /dev/ttyACM0 --id right_arm
    python -m live.calibrate --port /dev/ttyACM1 --id left_arm
    python -m live.calibrate --find-ports          # detect serial ports
"""

import argparse
import subprocess
import sys


def find_ports():
    """Run lerobot-find-port to detect connected serial devices."""
    print("Detecting serial ports...")
    print("Follow the prompts: unplug and replug the USB cable when asked.\n")
    try:
        subprocess.run(["lerobot-find-port"], check=True)
    except FileNotFoundError:
        print("Error: lerobot-find-port not found.")
        print("Install LeRobot first: pip install 'lerobot[feetech]'")
        sys.exit(1)


def calibrate(port: str, arm_id: str):
    """Run lerobot-calibrate for one arm."""
    print(f"Calibrating arm '{arm_id}' on {port}")
    print()
    print("=" * 60)
    print("CALIBRATION INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. The arm should be powered on and connected via USB.")
    print("2. When prompted, move ALL joints to their MIDPOINT")
    print("   (roughly the middle of their range of motion).")
    print("3. Press Enter.")
    print("4. Then move each joint through its FULL range of motion.")
    print("5. Press Enter when done.")
    print()
    print("The calibration file will be saved automatically.")
    print("=" * 60)
    print()

    cmd = [
        "lerobot-calibrate",
        f"--robot.type=so101_follower",
        f"--robot.port={port}",
        f"--robot.id={arm_id}",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: lerobot-calibrate not found.")
        print("Install LeRobot first: pip install 'lerobot[feetech]'")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Calibration failed with exit code {e.returncode}")
        sys.exit(1)

    print(f"\nCalibration for '{arm_id}' complete.")


def verify(port: str):
    """Quick verification: connect, read positions, disconnect."""
    print(f"\nVerifying connection on {port}...")
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        config = SOFollowerRobotConfig(
            port=port,
            use_degrees=True,
            disable_torque_on_disconnect=True,
        )
        robot = SOFollower(config)
        robot.connect()
        obs = robot.get_observation()

        print("Current joint positions (degrees):")
        for key, val in sorted(obs.items()):
            if key.endswith(".pos"):
                name = key.replace(".pos", "")
                print(f"  {name:20s}: {val:+8.1f}")

        robot.disconnect()
        print("Verification OK.")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="SO-ARM101 calibration helper")
    p.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    p.add_argument("--id", type=str, default="follower", help="Arm identifier for calibration file")
    p.add_argument("--find-ports", action="store_true", help="Detect connected serial ports")
    p.add_argument("--verify-only", action="store_true", help="Skip calibration, just read positions")
    args = p.parse_args()

    if args.find_ports:
        find_ports()
        return

    if args.verify_only:
        verify(args.port)
        return

    calibrate(args.port, args.id)
    verify(args.port)


if __name__ == "__main__":
    main()
