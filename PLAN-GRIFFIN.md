# Griffin's Plan — Data Engineer

**Role:** Camera capture pipeline + project configuration
**Files to create:** `config.py`, `camera.py`, `requirements.txt`, `.gitignore`

---

## What You Own

You build the **data ingestion layer** — the camera capture system and the shared configuration that every other module imports. Your code is the first thing that runs in the pipeline. Every frame the team processes starts with your `Camera.read()` call.

---

## File 1: `config.py`

**Purpose:** Single source of truth for all constants. Every team member imports from here.

### What to implement

```python
"""
config.py — Shared configuration constants for the POV teleop pipeline.

All tunable parameters live here. No other file should hardcode these values.
"""

import numpy as np

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_DEVICE = 0             # OpenCV device index (0 = default webcam)
CAMERA_WIDTH = 1280           # Capture resolution width
CAMERA_HEIGHT = 720           # Capture resolution height
CAMERA_FPS = 30               # Target framerate
ENABLE_UNDISTORT = False      # Set True for GoPro fish-eye correction

# GoPro fish-eye correction coefficients (approximate Hero wide-angle)
# Calibrate with checkerboard for precision; these are reasonable defaults
GOPRO_CAMERA_MATRIX = np.array([
    [910.0, 0.0, 640.0],
    [0.0, 910.0, 360.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

GOPRO_DIST_COEFFS = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

# ──────────────────────────────────────────────
# MediaPipe Models
# ──────────────────────────────────────────────
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

# ──────────────────────────────────────────────
# Pose Estimation Thresholds
# ──────────────────────────────────────────────
MIN_POSE_CONFIDENCE = 0.5     # Minimum detection confidence for pose
MIN_TRACKING_CONFIDENCE = 0.5 # Frame-to-frame tracking threshold
MIN_VISIBILITY = 0.5          # Minimum per-landmark visibility to trust

# Landmark indices (MediaPipe Pose, 33-landmark model)
# Right arm chain
LM_RIGHT_SHOULDER = 12
LM_RIGHT_ELBOW = 14
LM_RIGHT_WRIST = 16
LM_RIGHT_INDEX = 20
LM_RIGHT_THUMB = 22
LM_RIGHT_PINKY = 18
LM_RIGHT_HIP = 24

# Left arm chain (for future use)
LM_LEFT_SHOULDER = 11
LM_LEFT_ELBOW = 13
LM_LEFT_WRIST = 15
LM_LEFT_INDEX = 19
LM_LEFT_HIP = 23

# Which arm to track
TRACK_RIGHT_ARM = True

# Required landmarks that must be visible for arm tracking
REQUIRED_LANDMARKS = [LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST, LM_RIGHT_HIP]

# ──────────────────────────────────────────────
# Joint Limits (from actual SO-ARM101 MJCF — do NOT change these)
# ──────────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_LIMITS = {
    "shoulder_pan":  (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex":    (-1.690, 1.690),
    "wrist_flex":    (-1.658, 1.658),
    "wrist_roll":    (-2.744, 2.841),
    "gripper":       (-0.175, 1.745),
}

# Home position (from MJCF keyframe)
HOME_POSITION = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

# ──────────────────────────────────────────────
# Smoothing
# ──────────────────────────────────────────────
SMOOTHING_ALPHA = 0.3         # EMA alpha: 0.1=smooth/sluggish, 0.3=balanced, 0.6=responsive
DEADBAND_THRESHOLD = 0.02     # Ignore joint changes smaller than this (radians)

# ──────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────
SIM_RENDER_MODE = "human"
SIM_OBS_TYPE = "pixels_agent_pos"
SIM_CAMERA_CONFIG = "front_wrist"

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
DISPLAY_WIDTH = 1280          # Total display window width
DISPLAY_HEIGHT = 520          # Total display window height (480 video + 40 dashboard)
WINDOW_NAME = "POV Teleop"
```

### Implementation notes

- Every constant that could ever need tuning goes here
- Other files do `from config import CAMERA_WIDTH, JOINT_LIMITS, ...` etc.
- The joint limits are from the actual SO-ARM101 MuJoCo model (`so101_new_calib.xml`) — do not approximate these

---

## File 2: `camera.py`

**Purpose:** Wraps OpenCV video capture. Handles device selection, resolution, GoPro fish-eye correction.

### What to implement

```python
"""
camera.py — Camera capture with optional GoPro fish-eye correction.

Usage:
    cam = Camera(device=0, undistort=False)
    ok, frame = cam.read()  # frame is BGR np.ndarray (H, W, 3) or None
    cam.release()
"""

import cv2
import numpy as np
from config import (
    CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    ENABLE_UNDISTORT, GOPRO_CAMERA_MATRIX, GOPRO_DIST_COEFFS,
)


class Camera:
    """OpenCV video capture wrapper with optional fish-eye undistortion."""

    def __init__(self, device=CAMERA_DEVICE, width=CAMERA_WIDTH,
                 height=CAMERA_HEIGHT, fps=CAMERA_FPS, undistort=ENABLE_UNDISTORT):
        # Store params
        # Open cv2.VideoCapture(device)
        # Set CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
        # If undistort=True, pre-compute remap tables using cv2.initUndistortRectifyMap
        #   with GOPRO_CAMERA_MATRIX and GOPRO_DIST_COEFFS
        #   Store map1, map2 as instance variables
        # Raise RuntimeError if camera fails to open
        ...

    def read(self):
        """
        Read a single frame from the camera.

        Returns:
            tuple: (success: bool, frame: np.ndarray | None)
                   frame is BGR, shape (H, W, 3), dtype uint8
                   If undistort is enabled, frame is already corrected
        """
        # cap.read() → (ret, frame)
        # If undistort enabled and ret: frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        # Return (ret, frame)
        ...

    def release(self):
        """Release the camera device."""
        # cap.release()
        ...


def detect_cameras(max_index=5):
    """
    Utility: probe camera indices 0..max_index, return list of working indices.
    Useful for finding the right device when multiple cameras are connected.
    """
    # Loop through indices, try cv2.VideoCapture(i), check isOpened(), release
    # Return list of valid indices
    ...
```

### Implementation details

**Fish-eye undistortion approach:**
1. At `__init__` time (once), call `cv2.getOptimalNewCameraMatrix()` and `cv2.initUndistortRectifyMap()` to pre-compute the remap tables
2. At `read()` time (every frame), call `cv2.remap()` with the pre-computed tables
3. This costs ~2ms per frame at 720p — negligible
4. The pre-computation avoids the 20ms+ cost of `cv2.undistort()` per frame

**Camera auto-detection:**
- `detect_cameras()` is a helper for when GoPro/phone is plugged in and you don't know the device index
- Probe indices 0-5, return which ones open successfully

### Standalone test (put at bottom of file)

```python
if __name__ == "__main__":
    import time
    cam = Camera()
    print(f"Camera opened: device={cam.device}")

    # Also run detect_cameras and print results
    print(f"Available cameras: {detect_cameras()}")

    fps_counter = 0
    t0 = time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        fps_counter += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"FPS: {fps_counter / elapsed:.1f}")
            fps_counter = 0
            t0 = time.time()

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
```

---

## File 3: `requirements.txt`

```
mediapipe>=0.10.30
opencv-python>=4.8.0
numpy>=1.24.0
gym-soarm>=0.4.0
scipy>=1.11.0
```

---

## File 4: `.gitignore`

```
.venv/
__pycache__/
*.pyc
*.pyo
models/*.task
assets/*.mp4
.DS_Store
*.egg-info/
dist/
build/
```

---

## Acceptance Criteria

Your code is done when:

1. `python config.py` imports without errors (no syntax issues)
2. `python camera.py` opens the webcam, displays frames in a window, prints FPS
3. Pressing `q` cleanly closes the camera window
4. `detect_cameras()` correctly lists available camera indices
5. If `undistort=True` is passed, frames are corrected without crashing (even if the coefficients aren't perfectly calibrated)
6. Every constant from the master plan's interface contracts is present in `config.py`

---

## How Your Code Connects to Others

```
Your camera.py                    Torrin's pose.py
─────────────                    ──────────────
cam = Camera()                   tracker = PoseTracker()
ok, frame = cam.read()     →    result = tracker.process(frame, timestamp_ms)
                                 # frame is the BGR np.ndarray you produced
```

Torrin's code will call `cam.read()` (via main.py) and pass the frame directly to `tracker.process()`. The frame must be:
- BGR color order (OpenCV default)
- dtype uint8
- shape (H, W, 3) where H and W match CAMERA_HEIGHT and CAMERA_WIDTH

That's it. You don't need to know anything about MediaPipe, joint angles, or the simulator.
