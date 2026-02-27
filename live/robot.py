"""
robot.py -- Real SO-ARM101 robot controller via LeRobot.

Drop-in replacement for src/sim.py. One instance per arm.

Uses calibration files from the robo-ops repo (robo-ops/calibration/follower/).

Usage:
    robot = RobotController(port="/dev/ttyACM0", arm="right")
    robot.connect()
    robot.home(home_position_rad)
    robot.step(action_rad_6d)
    actual = robot.get_joint_positions()
    robot.disconnect()
"""

import os
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig, SOFollowerConfig

from convert import radians_to_lerobot_action, lerobot_obs_to_radians

# Calibration from the robo-ops repo cloned alongside live/
_PROJECT_ROOT = Path(__file__).parent.parent
_ROBO_OPS_CALIBRATION = _PROJECT_ROOT / "robo-ops" / "calibration" / "follower"


class RobotController:
    """
    Controls a single SO-ARM101 via LeRobot's SO101Follower.

    Accepts 6D actions in radians (same interface as the pipeline output),
    converts to degrees internally before sending to hardware.

    Uses calibration files from robo-ops/calibration/follower/ by default.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        max_relative_target: float = 5.0,
        arm: str = "right",
        label: str | None = None,
        calibration_dir: Path | None = None,
    ):
        self._label = label or arm
        cal_dir = calibration_dir or _ROBO_OPS_CALIBRATION
        cal_id = f"bimanual_follower_{arm}"

        if not cal_dir.exists():
            print(f"[{self._label}] WARNING: Calibration dir not found: {cal_dir}")
            print(f"[{self._label}] Clone robo-ops or run lerobot-calibrate first.")

        self._config = SO101FollowerConfig(
            port=port,
            max_relative_target=max_relative_target,
            use_degrees=True,
            disable_torque_on_disconnect=True,
            id=cal_id,
            calibration_dir=cal_dir,
        )
        self._robot = SO101Follower(self._config)
        self._connected = False
        self._last_action_rad: np.ndarray | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect to the robot hardware."""
        self._robot.connect()
        self._connected = True
        print(f"[{self._label}] Connected on {self._config.port}")

    def disconnect(self) -> None:
        """Disconnect and disable torque."""
        if self._connected:
            try:
                self._robot.disconnect()
            except Exception as e:
                print(f"[{self._label}] Disconnect error: {e}")
            self._connected = False
            print(f"[{self._label}] Disconnected (torque disabled)")

    def get_joint_positions(self) -> np.ndarray:
        """
        Read current servo positions.

        Returns:
            np.ndarray shape (6,) in radians.
        """
        obs = self._robot.get_observation()
        return lerobot_obs_to_radians(obs)

    def step(self, action_rad: np.ndarray) -> None:
        """
        Send a 6D action in radians to the robot.

        LeRobot's max_relative_target limits how far the robot moves per call.
        """
        action_rad = np.asarray(action_rad, dtype=np.float64).flatten()[:6]
        lerobot_action = radians_to_lerobot_action(action_rad)
        self._robot.send_action(lerobot_action)
        self._last_action_rad = action_rad.copy()

    def home(
        self,
        home_position_rad: np.ndarray,
        duration_s: float = 2.0,
        steps: int = 60,
    ) -> None:
        """
        Smoothly interpolate from current position to home.

        Reads the current position first to avoid a dangerous jump.
        """
        current_rad = self.get_joint_positions()
        dt = duration_s / steps

        print(
            f"[{self._label}] Homing: "
            f"{np.round(np.degrees(current_rad), 1)} -> "
            f"{np.round(np.degrees(home_position_rad), 1)} deg "
            f"over {duration_s:.1f}s"
        )

        for i in range(1, steps + 1):
            alpha = i / steps
            interpolated = (1.0 - alpha) * current_rad + alpha * home_position_rad
            self.step(interpolated)
            time.sleep(dt)

        print(f"[{self._label}] Homing complete.")

    def freeze(self) -> None:
        """Hold current position by re-sending the last command."""
        if self._last_action_rad is not None:
            self.step(self._last_action_rad)
