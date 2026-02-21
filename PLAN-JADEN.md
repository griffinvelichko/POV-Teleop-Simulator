# Jaden's Plan — Software Engineer

**Role:** Simulator integration, visualization, main loop orchestration
**Files to create:** `sim.py`, `display.py`, `main.py`

---

## What You Own

You build the **integration and output layer** — the simulator wrapper, the visualization system, and the main loop that ties everyone's modules together. You are the final assembler. Your `main.py` imports from Griffin, Torrin, and Damian and orchestrates the full pipeline.

```
Griffin → Torrin → Damian → YOUR sim.py + display.py, orchestrated by YOUR main.py
```

---

## File 1: `sim.py`

**Purpose:** Wraps the `gym-soarm` MuJoCo simulator. Provides a clean interface to step the simulation and retrieve rendered frames.

### What to implement

```python
"""
sim.py — gym-soarm SO-ARM101 simulator wrapper.

Wraps the SoArm-v0 Gymnasium environment for use in the teleop pipeline.

Usage:
    sim = SimController()
    sim.reset()
    obs = sim.step(action)  # action is np.ndarray(6)
    frame = sim.get_sim_frame()  # BGR np.ndarray for display
    sim.close()
"""

import numpy as np
import gymnasium as gym
import gym_soarm  # noqa: F401 — registers the SoArm-v0 environment
import cv2

from config import SIM_RENDER_MODE, SIM_OBS_TYPE, SIM_CAMERA_CONFIG, HOME_POSITION


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

    def __init__(self, render_mode=SIM_RENDER_MODE, obs_type=SIM_OBS_TYPE,
                 camera_config=SIM_CAMERA_CONFIG):
        # Create the gym environment:
        # self._env = gym.make(
        #     "SoArm-v0",
        #     render_mode=render_mode,
        #     obs_type=obs_type,
        #     camera_config=camera_config,
        # )
        #
        # Store the last observation for frame retrieval
        # self._last_obs = None
        # self._last_action = None
        ...

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            dict: initial observation
        """
        # obs, info = self._env.reset()
        # self._last_obs = obs
        # return obs
        ...

    def step(self, action):
        """
        Step the simulation with a 6D action array.

        Args:
            action: np.ndarray of shape (6,) — joint angle targets in radians

        Returns:
            dict: observation from the environment
        """
        # Ensure action is the right shape and type
        # action = np.asarray(action, dtype=np.float64).flatten()[:6]
        #
        # obs, reward, terminated, truncated, info = self._env.step(action)
        # self._last_obs = obs
        # self._last_action = action
        #
        # If terminated or truncated, auto-reset:
        # if terminated or truncated:
        #     obs, info = self._env.reset()
        #     self._last_obs = obs
        #
        # return obs
        ...

    def get_sim_frame(self):
        """
        Extract the latest simulator camera image as a BGR frame.

        Returns:
            np.ndarray: BGR image suitable for cv2.imshow, or None if unavailable
        """
        # The observation dict (when obs_type="pixels_agent_pos") contains:
        #   obs["pixels"]["front"] → np.ndarray (480, 640, 3) RGB
        #   obs["pixels"]["wrist"] → np.ndarray (480, 640, 3) RGB  (if camera_config includes wrist)
        #
        # Extract the front camera image, convert RGB → BGR for OpenCV:
        # if self._last_obs is not None and "pixels" in self._last_obs:
        #     rgb = self._last_obs["pixels"]["front"]
        #     return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # return None
        ...

    def get_joint_positions(self):
        """
        Get the current joint positions from the simulator.

        Returns:
            np.ndarray of shape (6,) or None
        """
        # if self._last_obs is not None and "agent_pos" in self._last_obs:
        #     return self._last_obs["agent_pos"]
        # return None
        ...

    def close(self):
        """Close the Gymnasium environment."""
        # self._env.close()
        ...
```

### Important: gym-soarm observation structure

When `obs_type="pixels_agent_pos"`, the observation is a dict:

```python
{
    "pixels": {
        "front": np.ndarray (480, 640, 3),  # RGB uint8
        "wrist": np.ndarray (480, 640, 3),  # RGB uint8 (if camera_config includes wrist)
    },
    "agent_pos": np.ndarray (6,),  # current joint positions in radians
}
```

When `render_mode="human"`, gym-soarm also opens its own OpenCV window. You can either:
- Use `render_mode="human"` for the built-in viewer AND extract frames from observations
- Use `render_mode=None` for headless, and only use your own display window

**Recommendation:** Start with `render_mode="human"` to verify the sim works, then switch to `render_mode=None` if you want full control over the display window via `display.py`.

### Standalone test

```python
if __name__ == "__main__":
    """Test simulator with random actions."""
    import time

    sim = SimController()
    obs = sim.reset()

    print(f"Action space: {sim._env.action_space}")
    print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")

    for i in range(200):
        # Random action within action space
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

    sim.close()
    cv2.destroyAllWindows()
```

---

## File 2: `display.py`

**Purpose:** Build the split-screen visualization with camera feed, simulator view, and joint angle dashboard.

### Target layout

```
┌──────────────────────────┬──────────────────────────┐
│                          │                          │
│   CAMERA FEED            │   SIMULATOR VIEW         │
│   (with skeleton overlay)│   (MuJoCo front camera)  │
│                          │                          │
│   640 x 480              │   640 x 480              │
│                          │                          │
└──────────────────────────┴──────────────────────────┘
│  Pan: +0.12  Lift: -0.45  Elbow: +1.02  ...  FPS:28│
└─────────────────────────────────────────────────────┘
```

### What to implement

```python
"""
display.py — Split-screen visualization with joint dashboard.

Composites the camera feed and simulator view side-by-side,
with a status bar showing joint angles and FPS.
"""

import cv2
import numpy as np
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, WINDOW_NAME, JOINT_NAMES


class Display:
    """
    Split-screen display builder.

    Usage:
        disp = Display()
        frame = disp.render(camera_frame, sim_frame, action, fps)
        keep_going = disp.show(frame)
        disp.close()
    """

    def __init__(self, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
        self.width = width
        self.height = height
        # Panel dimensions: each panel is half the width, full height minus dashboard
        self._dashboard_height = 40
        self._panel_height = height - self._dashboard_height
        self._panel_width = width // 2

    def render(self, camera_frame, sim_frame=None, action=None, fps=0.0):
        """
        Build the composited display frame.

        Args:
            camera_frame: BGR np.ndarray from camera (with skeleton overlay drawn)
            sim_frame: BGR np.ndarray from simulator, or None (shows placeholder)
            action: np.ndarray(6) current action, or None
            fps: float, current pipeline FPS

        Returns:
            np.ndarray: BGR composited frame ready for cv2.imshow
        """
        # 1. Resize camera_frame to panel size
        # left_panel = cv2.resize(camera_frame, (self._panel_width, self._panel_height))
        ...

        # 2. Resize sim_frame to panel size (or create black placeholder if None)
        # if sim_frame is not None:
        #     right_panel = cv2.resize(sim_frame, (self._panel_width, self._panel_height))
        # else:
        #     right_panel = np.zeros((self._panel_height, self._panel_width, 3), dtype=np.uint8)
        #     cv2.putText(right_panel, "Waiting for sim...",
        #                 (self._panel_width // 4, self._panel_height // 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        ...

        # 3. Add labels to panels
        # cv2.putText(left_panel, "CAMERA", (10, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(right_panel, "SIMULATOR", (10, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ...

        # 4. Concatenate panels side by side
        # top = np.hstack([left_panel, right_panel])
        ...

        # 5. Build dashboard bar
        # dashboard = np.zeros((self._dashboard_height, self.width, 3), dtype=np.uint8)
        # dashboard[:] = (40, 40, 40)  # dark gray background
        #
        # If action is not None, draw each joint value:
        #   spacing = self.width // 7  (6 joints + FPS)
        #   for i, name in enumerate(JOINT_NAMES):
        #       text = f"{name[:5]}: {action[i]:+.2f}"
        #       x = 10 + i * spacing
        #       cv2.putText(dashboard, text, (x, 28),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        #
        # Draw FPS on the right:
        #   cv2.putText(dashboard, f"FPS: {fps:.0f}",
        #               (self.width - 100, 28),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ...

        # 6. Stack panels and dashboard vertically
        # frame = np.vstack([top, dashboard])
        # return frame
        ...

    def show(self, frame):
        """
        Display the frame in a named window.

        Args:
            frame: BGR np.ndarray from render()

        Returns:
            bool: True to continue, False if user pressed 'q' or ESC
        """
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and key != 27  # q or ESC

    def close(self):
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
```

### Standalone test

```python
if __name__ == "__main__":
    """Test display with synthetic data."""
    disp = Display()

    for i in range(300):
        # Fake camera frame (gradient)
        cam = np.zeros((480, 640, 3), dtype=np.uint8)
        cam[:, :, 1] = 100  # greenish
        cv2.putText(cam, "Fake Camera Feed", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Fake sim frame (blueish)
        sim = np.zeros((480, 640, 3), dtype=np.uint8)
        sim[:, :, 0] = 100
        cv2.putText(sim, "Fake Sim View", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Fake action (oscillating)
        t = i / 30.0
        action = np.array([
            np.sin(t) * 0.5,
            np.cos(t) * 0.3,
            np.sin(t * 2) * 0.8,
            np.cos(t * 0.5) * 0.2,
            0.0,
            0.5 + 0.3 * np.sin(t),
        ])

        frame = disp.render(cam, sim, action, fps=30.0)
        if not disp.show(frame):
            break

    disp.close()
```

---

## File 3: `main.py`

**Purpose:** The entry point. Orchestrates the full pipeline: camera → pose → mapping → smoothing → sim → display.

### What to implement

```python
"""
main.py — POV Teleop main loop.

Orchestrates: Camera → PoseTracker → JointMapper → Smoother → SimController → Display

Usage:
    python main.py                        # default webcam
    python main.py --camera 1             # specific camera index
    python main.py --camera 0 --undistort # GoPro with fish-eye correction
    python main.py --no-sim               # camera + pose only (no simulator)
    python main.py --no-hands             # skip hand detection (no gripper)
"""

import argparse
import time
import numpy as np

from config import (
    CAMERA_DEVICE, ENABLE_UNDISTORT, SMOOTHING_ALPHA,
    HOME_POSITION, JOINT_NAMES,
)
from camera import Camera
from pose import PoseTracker, draw_landmarks_on_frame
from mapping import JointMapper
from smoother import Smoother
from sim import SimController
from display import Display


def parse_args():
    parser = argparse.ArgumentParser(description="POV Teleop: Body-cam → SO-ARM101 simulator")
    parser.add_argument("--camera", type=int, default=CAMERA_DEVICE,
                        help="Camera device index (default: 0)")
    parser.add_argument("--undistort", action="store_true", default=ENABLE_UNDISTORT,
                        help="Enable GoPro fish-eye correction")
    parser.add_argument("--alpha", type=float, default=SMOOTHING_ALPHA,
                        help="Smoothing alpha (0.1=smooth, 0.3=balanced, 0.6=responsive)")
    parser.add_argument("--no-sim", action="store_true",
                        help="Run camera + pose only, no simulator")
    parser.add_argument("--no-hands", action="store_true",
                        help="Skip hand detection (disables gripper)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Initialize all modules ──
    print("Initializing camera...")
    camera = Camera(device=args.camera, undistort=args.undistort)

    print("Initializing pose tracker...")
    tracker = PoseTracker()
    # NOTE: if --no-hands, you could skip hand model loading or just ignore hand results

    print("Initializing joint mapper and smoother...")
    mapper = JointMapper()
    smoother = Smoother(alpha=args.alpha)

    sim = None
    if not args.no_sim:
        print("Initializing simulator...")
        sim = SimController()
        sim.reset()

    print("Initializing display...")
    display = Display()

    # ── State ──
    last_action = HOME_POSITION.copy()  # hold home position until first detection
    frame_count = 0
    t_start = time.time()
    fps = 0.0

    print("Pipeline ready. Move your right arm in front of the camera.")
    print("Press 'q' to quit.\n")

    # ── Main loop ──
    try:
        while True:
            # 1. Capture frame
            ok, frame = camera.read()
            if not ok:
                print("Camera read failed.")
                break

            # 2. Compute timestamp (monotonically increasing, milliseconds)
            timestamp_ms = int((time.time() - t_start) * 1000)

            # 3. Run pose estimation
            pose_result = tracker.process(frame, timestamp_ms)

            # 4. Extract joint angles
            raw_action = mapper.compute(pose_result)

            if raw_action is not None:
                # New valid detection → smooth and use
                action = smoother.update(raw_action)
                last_action = action
            else:
                # Arm not visible → hold last known position
                action = last_action

            # 5. Step simulator
            sim_frame = None
            if sim is not None and action is not None:
                sim.step(action)
                sim_frame = sim.get_sim_frame()

            # 6. Draw skeleton overlay on camera frame
            draw_landmarks_on_frame(frame, pose_result)

            # 7. Compute FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed > 0:
                fps = frame_count / elapsed

            # 8. Build and show display
            display_frame = display.render(frame, sim_frame, action, fps)
            if not display.show(display_frame):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # ── Cleanup ──
    print("Shutting down...")
    camera.release()
    tracker.close()
    if sim is not None:
        sim.close()
    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
```

### CLI usage

```bash
# Default: webcam + full pipeline
python main.py

# Specific camera (e.g. GoPro on index 1)
python main.py --camera 1

# GoPro with fish-eye correction
python main.py --camera 1 --undistort

# Camera + pose only (no simulator window) — for testing tracking
python main.py --no-sim

# More responsive (less smoothing)
python main.py --alpha 0.5

# Very smooth (good for demo)
python main.py --alpha 0.15
```

---

## Key Implementation Details

### 1. gym-soarm API

The package is confirmed real and on PyPI (v0.4.0). Key facts:

```python
import gymnasium as gym
import gym_soarm  # This import REGISTERS the environment

env = gym.make("SoArm-v0",
    render_mode="human",          # opens OpenCV viewer
    obs_type="pixels_agent_pos",  # returns camera images + joint positions
    camera_config="front_wrist",  # front + wrist cameras
)

# Action space: Box(6,) — 6 continuous values (joint angles in radians)
# Observation: dict with "pixels" (dict of camera images) and "agent_pos" (6D array)
```

**Keyboard controls in human render mode:** Press `1`/`2`/`3` to switch camera views, `q` or `ESC` to quit.

### 2. Render Mode Decision

`render_mode="human"` opens gym-soarm's own OpenCV window. This means you'll have TWO windows:
- gym-soarm's built-in viewer
- Your `display.py` split-screen window

**Options:**
- **Option A (simpler):** Use `render_mode="human"` for dev. Two windows is fine for hackathon.
- **Option B (cleaner):** Use `render_mode=None` (headless). Extract frames from observations only. Your single `display.py` window shows everything.

Start with Option A. Switch to B if the two-window setup is confusing for the demo.

### 3. Frame Extraction from Observations

When `obs_type="pixels_agent_pos"`:

```python
obs["pixels"]["front"]  # np.ndarray (480, 640, 3) RGB
obs["pixels"]["wrist"]  # np.ndarray (480, 640, 3) RGB
obs["agent_pos"]        # np.ndarray (6,) — current joint positions
```

The images are **RGB** — convert to **BGR** for OpenCV:
```python
bgr = cv2.cvtColor(obs["pixels"]["front"], cv2.COLOR_RGB2BGR)
```

### 4. Display Timing

`cv2.waitKey(1)` in `display.show()` gives ~1ms wait. The pipeline will naturally run at whatever FPS the slowest component allows (usually MediaPipe at ~30 FPS). Don't add artificial delays.

### 5. The "Hold Position" Pattern

When the arm leaves the camera's field of view, `mapper.compute()` returns `None`. The main loop holds `last_action` — the last valid action — and keeps sending it to the simulator. This prevents the robot arm from snapping to a default position when tracking is temporarily lost.

### 6. Fallback: If gym-soarm Won't Install

If `pip install gym-soarm` fails (e.g., MuJoCo binary issues), fall back to raw MuJoCo:

```bash
pip install mujoco
```

Then the SO-ARM100 model is in MuJoCo Menagerie at `trs_so_arm100/scene.xml`. You'd write a minimal step loop:

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("path/to/scene.xml")
data = mujoco.MjData(model)
data.ctrl[:] = action  # set joint targets
mujoco.mj_step(model, data)
```

But this is the fallback, not the plan.

---

## Acceptance Criteria

1. `python sim.py` opens a MuJoCo window and shows the robot arm moving randomly
2. `python display.py` shows a split-screen window with fake data and a dashboard bar
3. `python main.py` runs the full pipeline: camera → skeleton overlay → joint angles → sim → display
4. `python main.py --no-sim` runs camera + pose only (useful for testing without MuJoCo)
5. Pressing `q` cleanly shuts down all windows and releases all resources
6. FPS is displayed correctly and stays above 15
7. The robot arm in the simulator responds to human arm movements

---

## How Your Code Connects to Everyone Else's

```
Griffin's camera.py  → provides BGR frames
Torrin's pose.py     → provides PoseResult (landmarks)
Damian's mapping.py  → provides 6D action arrays
Damian's smoother.py → smooths the action arrays

YOUR sim.py          → consumes 6D action arrays, produces sim frames
YOUR display.py      → composites camera frame + sim frame + dashboard
YOUR main.py         → imports ALL modules, runs the loop
```

You are the integrator. If anything doesn't connect, you're the person who debugs the interface boundaries. The contract signatures in `PLAN-MASTER.md` are your reference.
