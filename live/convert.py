"""
convert.py -- Unit conversion between pipeline radians and LeRobot degrees.

The src/ pipeline outputs joint angles in radians. LeRobot expects degrees
for the 5 revolute joints and a 0-100 percentage for the gripper.
"""

import numpy as np

# Gripper range in radians (from src/config.py JOINT_LIMITS)
_GRIPPER_RAD_MIN = -0.175
_GRIPPER_RAD_MAX = 1.745
_GRIPPER_RAD_RANGE = _GRIPPER_RAD_MAX - _GRIPPER_RAD_MIN

# LeRobot motor names (no "left_" prefix -- both arms use the same motor names
# because each arm is a separate SOFollower on its own serial port)
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def radians_to_lerobot_action(action_rad: np.ndarray) -> dict[str, float]:
    """
    Convert 6D action (radians) to LeRobot send_action dict.

    Revolute joints -> degrees. Gripper -> 0-100 percentage.

    Args:
        action_rad: shape (6,) [pan, lift, elbow, wrist, roll, gripper] in radians.

    Returns:
        dict with keys like "shoulder_pan.pos": float.
    """
    result = {}
    for i, name in enumerate(MOTOR_NAMES):
        if name == "gripper":
            normalized = (action_rad[i] - _GRIPPER_RAD_MIN) / _GRIPPER_RAD_RANGE
            result[f"{name}.pos"] = float(np.clip(normalized, 0.0, 1.0)) * 100.0
        else:
            result[f"{name}.pos"] = float(np.degrees(action_rad[i]))
    return result


def lerobot_obs_to_radians(obs: dict) -> np.ndarray:
    """
    Convert LeRobot observation dict (degrees) back to radians.

    Args:
        obs: dict with keys like "shoulder_pan.pos": float.

    Returns:
        np.ndarray shape (6,) in radians.
    """
    action = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(MOTOR_NAMES):
        key = f"{name}.pos"
        if key not in obs:
            continue
        if name == "gripper":
            pct = obs[key]
            action[i] = _GRIPPER_RAD_MIN + (pct / 100.0) * _GRIPPER_RAD_RANGE
        else:
            action[i] = float(np.radians(obs[key]))
    return action
