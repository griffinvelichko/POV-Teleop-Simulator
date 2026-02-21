"""
sim.py — gym-soarm SO-ARM101 simulator wrapper.

Wraps the gym_soarm/PickAndPlaceCube-v0 Gymnasium environment for use in the teleop pipeline.

Usage:
    sim = SimController()
    sim.reset()
    obs = sim.step(action)  # action is np.ndarray(6)
    frame = sim.get_sim_frame()  # BGR np.ndarray for display
    sim.close()
"""

import os
import platform

import numpy as np
import cv2
import gymnasium as gym

from config import SIM_RENDER_MODE, SIM_OBS_TYPE, SIM_CAMERA_CONFIG


def _ensure_gym_soarm():
    """Import and register SoArm-v0. Deferred so --no-sim runs without torch/gym_soarm."""
    # On macOS: import torch first (while platform is Darwin) so it loads .dylib;
    # then patch platform so MuJoCo uses GLFW (Apple removed OpenGL.framework).
    _platform_system_orig = None
    if platform.system() == "Darwin":
        import torch  # noqa: F401 — load before patching so torch picks libtorch_global_deps.dylib
        os.environ.setdefault("MUJOCO_GL", "glfw")
        _platform_system_orig = platform.system
        platform.system = lambda: "Unknown"
    try:
        import gym_soarm  # noqa: F401 — registers the SoArm-v0 environment
    except OSError as e:
        raise RuntimeError(
            "Failed to load simulator (gym_soarm/torch/MuJoCo). "
            "Run with --no-sim for camera+pose only, or fix the env (e.g. reinstall torch)."
        ) from e
    finally:
        if _platform_system_orig is not None:
            platform.system = _platform_system_orig


class SimController:
    """
    Gymnasium environment wrapper for the SO-ARM101 MuJoCo simulation.

    The environment's action space is Box(6,) where:
      action[0] = shoulder_pan   (radians)
      action[1] = shoulder_lift  (radians)
      action[2] = elbow_flex     (radians)
      action[3] = wrist_flex     (radians)
      action[4] = wrist_roll     (radians)
      action[5] = gripper        (radians)
    """

    def __init__(
        self,
        render_mode=SIM_RENDER_MODE,
        obs_type=SIM_OBS_TYPE,
        camera_config=SIM_CAMERA_CONFIG,
    ):
        _ensure_gym_soarm()
        self._env = gym.make(
            "gym_soarm/PickAndPlaceCube-v0",
            render_mode=render_mode,
            obs_type=obs_type,
            camera_config=camera_config,
        )
        self._last_obs = None
        self._last_action = None

    def reset(self) -> dict:
        """
        Reset the environment to initial state.

        Returns:
            dict: initial observation
        """
        obs, info = self._env.reset()
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> dict:
        """
        Step the simulation with a 6D action array.

        Args:
            action: np.ndarray of shape (6,) — joint angle targets in radians

        Returns:
            dict: observation from the environment
        """
        action = np.asarray(action, dtype=np.float64).flatten()[:6]
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_obs = obs
        self._last_action = action

        if terminated or truncated:
            obs, info = self._env.reset()
            self._last_obs = obs

        return obs

    def get_sim_frame(self) -> np.ndarray | None:
        """
        Extract the latest simulator camera image as a BGR frame.

        Returns:
            np.ndarray: BGR image suitable for cv2.imshow, or None if unavailable
        """
        if self._last_obs is None:
            return None
        obs = self._last_obs
        if isinstance(obs, dict) and "pixels" in obs:
            pixels = obs["pixels"]
            # gym_soarm uses "front_camera" for front_wrist config
            for key in ("front_camera", "front", "diagonal", "overview_camera"):
                if key in pixels:
                    rgb = pixels[key]
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if pixels:
                rgb = next(iter(pixels.values()))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return None

    def get_joint_positions(self) -> np.ndarray | None:
        """
        Get the current joint positions from the simulator.

        Returns:
            np.ndarray of shape (6,) or None
        """
        if self._last_obs is not None and isinstance(self._last_obs, dict):
            if "agent_pos" in self._last_obs:
                return np.asarray(self._last_obs["agent_pos"], dtype=np.float64)
        return None

    def close(self) -> None:
        """Close the Gymnasium environment."""
        self._env.close()


if __name__ == "__main__":
    """Test simulator with random actions."""
    import time

    sim = SimController()
    obs = sim.reset()

    print(f"Action space: {sim._env.action_space}")
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    try:
        for i in range(200):
            action = sim._env.action_space.sample()
            obs = sim.step(action)

            frame = sim.get_sim_frame()
            joints = sim.get_joint_positions()

            if frame is not None:
                cv2.imshow("Sim Test", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

            if joints is not None and i % 30 == 0:
                print(f"Step {i}: joints = {joints}")
    finally:
        sim.close()
        cv2.destroyAllWindows()
