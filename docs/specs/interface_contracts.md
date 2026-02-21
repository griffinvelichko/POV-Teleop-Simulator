# Interface Contracts

These are the exact function signatures each person must implement. **Do not deviate from these signatures** — they are how the modules connect.

## Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  GRIFFIN     │    │  TORRIN      │    │  DAMIAN      │    │  JADEN       │
│  camera.py  │───→│  pose.py     │───→│  mapping.py  │───→│  sim.py      │
│  config.py  │    │              │    │  smoother.py │    │  display.py  │
│             │    │              │    │              │    │  main.py     │
│  BGR frame  │    │  PoseResult  │    │  6D action   │    │  Sim + viz   │
│  np.ndarray │    │  dataclass   │    │  np.array(6) │    │  window      │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Interface 1: Griffin → Torrin (camera.py → pose.py)

```python
class Camera:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, undistort: bool = False) -> None: ...
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Returns (success, bgr_frame). Frame is np.ndarray shape (H, W, 3) dtype uint8, or None."""
        ...
    def release(self) -> None: ...
```

**Contract**:
- `read()` returns `(True, frame)` on success, `(False, None)` on failure
- Frame is always BGR color order, shape `(H, W, 3)`, dtype `uint8`
- Fish-eye correction (if enabled) is applied before returning — caller never sees distorted frames

---

## Interface 2: Torrin → Damian (pose.py → mapping.py)

```python
@dataclass
class PoseResult:
    pose_world_landmarks: list | None = None   # 33 world landmarks (meters, hip-centered)
    pose_landmarks: list | None = None          # 33 normalized landmarks (0-1, for drawing)
    hand_landmarks: list | None = None          # 21 hand landmarks (normalized)
    hand_world_landmarks: list | None = None    # 21 hand world landmarks (meters)
    timestamp_ms: int = 0

class PoseTracker:
    def __init__(self, model_path: str = "models/pose_landmarker_full.task",
                 hand_model_path: str = "models/hand_landmarker.task") -> None: ...
    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> PoseResult:
        """Process a BGR frame and return pose results."""
        ...
    def close(self) -> None: ...
```

**Contract**:
- `pose_world_landmarks` are **world landmarks** (meters, hip-centered) with `.x`, `.y`, `.z` attributes — use these for angle computation
- `pose_landmarks` are **normalized** (0-1) — use for drawing skeleton overlays only
- `hand_landmarks` are **normalized** (0-1) — use for gripper distance
- Visibility is accessed per-landmark via `landmark.visibility` (0.0-1.0) — check before computing angles
- All fields are `None` if no person detected in frame
- Right arm landmarks: shoulder=12, elbow=14, wrist=16, index_finger=20, thumb=22, hip=24

---

## Interface 3: Damian → Jaden (mapping.py + smoother.py → main.py)

```python
class JointMapper:
    def __init__(self) -> None: ...
    def compute(self, pose_result: PoseResult) -> np.ndarray | None:
        """
        Returns shape (6,): [shoulder_pan, shoulder_lift, elbow_flex,
                              wrist_flex, wrist_roll, gripper]
        All values in radians, clamped to JOINT_LIMITS from config.py.
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

**Contract**:
- `compute()` returns `np.ndarray` shape `(6,)` or `None`
- Joint order is fixed: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- All angles in radians, clamped to `JOINT_LIMITS` from `config.py`
- `None` means "landmarks not visible enough to compute" — caller should hold last action
- Smoother is stateful — call `reset()` to clear history

---

## Interface 4: Jaden's Modules (sim.py + display.py + main.py)

```python
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
    def __init__(self, width: int = 1280, height: int = 520) -> None: ...
    def render(self, camera_frame: np.ndarray, sim_frame: np.ndarray | None,
               action: np.ndarray | None, fps: float) -> np.ndarray:
        """Build the split-screen display. Returns composited BGR frame."""
        ...
    def show(self, frame: np.ndarray) -> bool:
        """Display frame in window. Returns False if user pressed 'q'."""
        ...
    def close(self) -> None: ...
```

**Contract**:
- `step()` accepts `np.ndarray` shape `(6,)` in radians
- `get_sim_frame()` returns BGR `np.ndarray` or `None`
- `show()` returns `False` when user presses 'q' — main loop should break
- Display renders split-screen: camera feed (left) | simulator view (right) | dashboard (bottom)

---

## Shared Constants (config.py)

Griffin writes `config.py`. Everyone imports from it. Key constants:

```python
# Joint limits from actual SO-ARM101 MJCF
JOINT_LIMITS = {
    "shoulder_pan":  (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex":    (-1.690, 1.690),
    "wrist_flex":    (-1.658, 1.658),
    "wrist_roll":    (-2.744, 2.841),
    "gripper":       (-0.175, 1.745),
}

# Smoothing
SMOOTHING_ALPHA = 0.3
DEADBAND_THRESHOLD = 0.02  # radians

# Camera
CAMERA_DEVICE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
ENABLE_UNDISTORT = False

# MediaPipe
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
MIN_VISIBILITY = 0.5

# Display
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 520
```

---

## Testing Strategy

Each person can test independently before integration:

| Person | Standalone Test | What to Verify |
|--------|----------------|----------------|
| Griffin | `python camera.py` | Opens webcam, shows frames, prints FPS |
| Torrin | `python pose.py` | Draws skeleton overlay on webcam feed |
| Damian | `python mapping.py` | Prints 6D action arrays from hardcoded landmarks |
| Jaden | `python sim.py` | Shows robot arm moving with random actions |

### Integration Tests (in order)

1. **Griffin + Torrin**: Camera feeds into pose tracker, skeleton overlay on screen
2. **Torrin + Damian**: Pose results feed into joint mapper, print action arrays
3. **Damian + Jaden**: Action arrays feed into simulator, robot moves
4. **Full pipeline**: `python main.py` — Camera → Pose → Mapper → Smoother → Sim → Display
