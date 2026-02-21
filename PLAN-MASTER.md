# POV Teleop — Master Implementation Plan

**Date:** February 21, 2026
**Team:** Griffin, Torrin, Damian, Jaden
**Goal:** Body-cam → pose estimation → simulated SO-ARM101 control, working demo

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GRIFFIN     │    │  TORRIN      │    │  DAMIAN      │    │  JADEN       │
│  camera.py  │───→│  pose.py     │───→│  mapping.py  │───→│  sim.py      │
│  config.py  │    │              │    │  smoother.py │    │  display.py  │
│             │    │              │    │              │    │  main.py     │
│  BGR frame  │    │  Landmarks   │    │  6D action   │    │  Sim + viz   │
│  np.ndarray │    │  dict/None   │    │  np.array(6) │    │  window      │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Data flows left to right. Each person owns their files. Interfaces are defined below.**

---

## File Ownership

| File | Owner | Purpose |
|------|-------|---------|
| `src/config.py` | Griffin | All shared constants (joint limits, camera params, thresholds) |
| `src/camera.py` | Griffin | Camera capture, frame output |
| `src/pose.py` | Torrin | MediaPipe Pose + Hands wrapper, landmark extraction |
| `src/mapping.py` | Damian | Landmark → joint angles → 6D robot action array |
| `src/smoother.py` | Damian | Exponential moving average filter |
| `src/sim.py` | Jaden | gym-soarm environment wrapper |
| `src/display.py` | Jaden | Split-screen visualization, overlays, dashboard |
| `src/main.py` | Jaden | Main loop orchestrating all modules |
| `requirements.txt` | Griffin | Python dependencies |

---

## Critical Dependency: MediaPipe API Change

**The legacy `mp.solutions.pose` API was REMOVED in mediapipe 0.10.30 (Dec 2025).** All code in the old IMPLEMENTATION-PLAN.md and RESEARCH.md uses the dead API.

**Decision: Use mediapipe 0.10.32 with the Tasks API.** This requires:
1. Downloading `.task` model files (not bundled in pip package)
2. Using `mp.tasks.vision.PoseLandmarker` instead of `mp.solutions.pose.Pose`
3. Different initialization and callback patterns

Torrin owns this complexity. The interface to Damian stays clean regardless.

---

## Shared Setup (Everyone Does This First)

```bash
# 1. Clone and enter project
cd /Users/gveli/Documents/pov-teleop

# 2. Verify Python version (MUST be 3.11 or 3.12 — mediapipe has no 3.13 support)
python3 --version

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Download MediaPipe model files
mkdir -p models
wget -O models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
wget -O models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# 6. Set MuJoCo renderer (macOS)
export MUJOCO_GL='glfw'
```

### requirements.txt

```
mediapipe>=0.10.30
opencv-python>=4.8.0
numpy>=1.24.0
gym-soarm>=0.4.0
scipy>=1.11.0
```

`gym-soarm` pulls in: `gymnasium>=0.29.1`, `mujoco>=2.3.7,<3.0.0`, `dm-control>=1.0.14`, `imageio[ffmpeg]>=2.34.0`

---

## Interface Contracts

These are the exact function signatures each person must implement. **Do not deviate from these signatures** — they are how the modules connect.

### Interface 1: Griffin → Torrin (camera.py → pose.py)

```python
# Griffin provides:
class Camera:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None: ...
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Returns (success, bgr_frame). Frame is np.ndarray shape (H, W, 3) dtype uint8, or None."""
        ...
    def release(self) -> None: ...

# Torrin consumes: bgr_frame (np.ndarray, shape (H, W, 3), dtype uint8, BGR color order)
```

### Interface 2: Torrin → Damian (pose.py → mapping.py)

```python
# Torrin provides:
@dataclass
class PoseResult:
    """Result from a single frame of pose estimation."""
    pose_landmarks: list | None          # 33 landmarks with x,y,z,visibility (world coords, meters)
    hand_landmarks: list | None          # 21 hand landmarks with x,y,z (normalized)
    pose_visibility: list[float] | None  # 33 visibility scores (0.0-1.0)
    timestamp_ms: int                    # Frame timestamp in milliseconds

class PoseTracker:
    def __init__(self, model_path: str = "models/pose_landmarker_full.task",
                 hand_model_path: str = "models/hand_landmarker.task") -> None: ...
    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> PoseResult:
        """Process a BGR frame and return pose results."""
        ...
    def close(self) -> None: ...

# Each landmark in pose_landmarks is an object with: .x, .y, .z (float, meters, hip-centered)
# Each landmark in hand_landmarks is an object with: .x, .y, .z (float, normalized 0-1)
```

### Interface 3: Damian → Jaden (mapping.py → sim.py / main.py)

```python
# Damian provides:
class JointMapper:
    def __init__(self) -> None: ...
    def compute(self, pose_result: PoseResult) -> np.ndarray | None:
        """
        Takes a PoseResult, returns a 6D action array or None if landmarks not visible.
        Returns: np.ndarray shape (6,) = [shoulder_pan, shoulder_lift, elbow_flex,
                                           wrist_flex, wrist_roll, gripper]
        All values in radians, clamped to joint limits.
        Returns None if required landmarks are not sufficiently visible.
        """
        ...

class Smoother:
    def __init__(self, alpha: float = 0.3, num_joints: int = 6) -> None: ...
    def update(self, values: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing. Returns smoothed array of same shape."""
        ...
    def reset(self) -> None: ...
```

### Interface 4: Jaden's integration (sim.py + display.py + main.py)

```python
# Jaden provides:
class SimController:
    def __init__(self, render_mode: str = "human",
                 obs_type: str = "pixels_agent_pos",
                 camera_config: str = "front_wrist") -> None: ...
    def reset(self) -> dict: ...
    def step(self, action: np.ndarray) -> dict:
        """Step sim with 6D action array. Returns observation dict."""
        ...
    def get_sim_frame(self) -> np.ndarray | None:
        """Return current sim render as BGR image, or None."""
        ...
    def close(self) -> None: ...

class Display:
    def __init__(self, width: int = 1280, height: int = 480) -> None: ...
    def render(self, camera_frame: np.ndarray, sim_frame: np.ndarray | None,
               action: np.ndarray | None, fps: float) -> np.ndarray:
        """Build the split-screen display. Returns composited BGR frame."""
        ...
    def show(self, frame: np.ndarray) -> bool:
        """Display frame in window. Returns False if user pressed 'q'."""
        ...
    def close(self) -> None: ...
```

---

## Integration Order (How the Pieces Connect)

### main.py pseudocode (Jaden writes this, uses everyone's modules)

```python
from config import *
from camera import Camera
from pose import PoseTracker
from mapping import JointMapper, Smoother
from sim import SimController
from display import Display

camera = Camera(device=CAMERA_DEVICE)
tracker = PoseTracker()
mapper = JointMapper()
smoother = Smoother(alpha=SMOOTHING_ALPHA)
sim = SimController()
display = Display()

sim.reset()
last_action = None

while True:
    ok, frame = camera.read()
    if not ok:
        break

    timestamp_ms = int(time.time() * 1000)
    pose_result = tracker.process(frame, timestamp_ms)
    raw_action = mapper.compute(pose_result)

    if raw_action is not None:
        action = smoother.update(raw_action)
        last_action = action
    else:
        action = last_action  # hold position when arm not visible

    if action is not None:
        obs = sim.step(action)
        sim_frame = sim.get_sim_frame()
    else:
        sim_frame = None

    display_frame = display.render(frame, sim_frame, action, fps)
    if not display.show(display_frame):
        break

# cleanup
camera.release()
tracker.close()
sim.close()
display.close()
```

---

## Testing Strategy

Each person can test independently before integration:

| Person | Standalone Test |
|--------|----------------|
| Griffin | Run `camera.py` directly — should open webcam and show frames in a window |
| Torrin | Run `pose.py` with a static image or webcam — should print landmark coordinates |
| Damian | Run `mapping.py` with hardcoded landmark values — should print 6D action arrays |
| Jaden | Run `sim.py` with random actions — should show robot arm moving randomly in MuJoCo |

### Integration Tests (in order)

1. **Griffin + Torrin:** Camera feeds into pose tracker, skeleton overlay on screen
2. **Torrin + Damian:** Pose results feed into joint mapper, print action arrays
3. **Damian + Jaden:** Action arrays feed into simulator, robot moves
4. **Full pipeline:** Camera → Pose → Mapper → Smoother → Sim → Display

---

## Corrected Joint Limits (from actual SO-ARM101 MJCF)

These go in `config.py` (Griffin writes, everyone imports):

```python
JOINT_LIMITS = {
    "shoulder_pan":  (-1.920, 1.920),   # rad, ~220 degrees
    "shoulder_lift": (-1.745, 1.745),   # rad, ~200 degrees
    "elbow_flex":    (-1.690, 1.690),   # rad, ~194 degrees
    "wrist_flex":    (-1.658, 1.658),   # rad, ~190 degrees
    "wrist_roll":    (-2.744, 2.841),   # rad, ~320 degrees
    "gripper":       (-0.175, 1.745),   # rad, ~110 degrees
}
```

---

## Milestone Checklist

- [x] **M1:** Each person's module runs standalone with its own `if __name__ == "__main__"` test
- [x] **M2:** Camera + Pose integration works (skeleton on screen)
- [x] **M3:** Pose + Mapping integration works (action arrays printed)
- [x] **M4:** Sim runs with hardcoded or random actions
- [x] **M5:** Full pipeline: move arm → robot moves in sim
- [x] **M6:** Gripper control via hand landmarks
- [x] **M7:** Split-screen visualization with dashboard
- [x] **M8:** Demo-ready with FPS counter and polish

---

## Fallback Plans

| Problem | Fallback | Owner |
|---------|----------|-------|
| MediaPipe Tasks API issues | Pin `mediapipe==0.10.21`, use legacy `mp.solutions.pose` | Torrin |
| gym-soarm won't install | Use raw MuJoCo with `trs_so_arm100/scene.xml` from Menagerie | Jaden |
| Camera not detected | Use `cv2.VideoCapture(0)` laptop webcam | Griffin |
| Gripper via hands unreliable | Map gripper to keyboard spacebar | Damian |
| Wrist roll too jittery | Lock to 0.0 (neutral) | Damian |
| Z-depth too noisy | Compute angles from x,y only (2D projection) | Damian |

---

## Individual Plans

Each team member has a detailed plan:

- **[PLAN-GRIFFIN.md](PLAN-GRIFFIN.md)** — Camera capture, config, project setup
- **[PLAN-TORRIN.md](PLAN-TORRIN.md)** — MediaPipe Pose + Hands (Tasks API)
- **[PLAN-DAMIAN.md](PLAN-DAMIAN.md)** — Joint mapping, smoothing, angle extraction
- **[PLAN-JADEN.md](PLAN-JADEN.md)** — Simulator, visualization, main loop integration
