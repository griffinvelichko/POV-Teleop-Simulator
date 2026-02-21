# Torrin's Plan — AI Engineer

**Role:** MediaPipe Pose + Hands integration (Tasks API)
**Files to create:** `pose.py`

---

## What You Own

You build the **AI inference layer** — the module that takes a raw camera frame and extracts human body landmarks from it. This is the most technically nuanced piece because the MediaPipe API recently changed (the old `mp.solutions` was removed in Dec 2025). You need to implement using the new **Tasks API**.

Your module sits between Griffin's camera output and Damian's joint angle extraction:
```
Griffin's frame (BGR np.ndarray) → YOUR pose.py → PoseResult → Damian's mapping.py
```

---

## Background: MediaPipe Tasks API

The old API (`mp.solutions.pose`) was removed in `mediapipe>=0.10.30`. The replacement is:

```python
import mediapipe as mp

# Classes you'll use:
mp.tasks.BaseOptions
mp.tasks.vision.PoseLandmarker
mp.tasks.vision.PoseLandmarkerOptions
mp.tasks.vision.HandLandmarker
mp.tasks.vision.HandLandmarkerOptions
mp.tasks.vision.RunningMode
mp.Image
```

**Key differences from the old API:**
1. You must download `.task` model files and pass the path at init
2. There are three running modes: `IMAGE`, `VIDEO`, `LIVE_STREAM`
3. `VIDEO` mode requires monotonically increasing timestamps
4. `LIVE_STREAM` mode uses async callbacks (don't use this — `VIDEO` mode is simpler and sufficient)
5. Results give you both `pose_landmarks` (normalized 0-1) and `pose_world_landmarks` (meters, hip-centered)
6. **Use `pose_world_landmarks`** for Damian's angle extraction — they're in meters and better for 3D geometry

---

## File: `pose.py`

### Data class for results

```python
"""
pose.py — MediaPipe Pose + Hand landmark extraction using the Tasks API.

Wraps PoseLandmarker and HandLandmarker into a single PoseTracker class.
Outputs a PoseResult dataclass consumed by mapping.py.

Requires model files:
  - models/pose_landmarker_full.task
  - models/hand_landmarker.task
"""

from dataclasses import dataclass, field
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions

from config import POSE_MODEL_PATH, HAND_MODEL_PATH, MIN_POSE_CONFIDENCE, MIN_TRACKING_CONFIDENCE


@dataclass
class PoseResult:
    """Result from a single frame of pose + hand estimation."""
    pose_world_landmarks: list | None = None
    """33 world landmarks (x, y, z in meters, origin at hip midpoint). None if no pose detected."""

    pose_landmarks: list | None = None
    """33 normalized landmarks (x, y in 0-1, z relative depth). None if no pose detected."""

    hand_landmarks: list | None = None
    """21 hand landmarks (normalized x, y, z). None if no hand detected."""

    hand_world_landmarks: list | None = None
    """21 hand world landmarks (meters). None if no hand detected."""

    timestamp_ms: int = 0
    """Frame timestamp in milliseconds."""
```

### PoseTracker class

```python
class PoseTracker:
    """
    Wraps MediaPipe PoseLandmarker + HandLandmarker.

    Usage:
        tracker = PoseTracker()
        result = tracker.process(bgr_frame, timestamp_ms)
        # result.pose_world_landmarks is a list of 33 landmarks or None
        tracker.close()
    """

    def __init__(self, model_path=POSE_MODEL_PATH, hand_model_path=HAND_MODEL_PATH):
        # ── Pose Landmarker ──
        # Create PoseLandmarkerOptions:
        #   base_options = BaseOptions(model_asset_path=model_path)
        #   running_mode = RunningMode.VIDEO
        #   min_pose_detection_confidence = MIN_POSE_CONFIDENCE
        #   min_tracking_confidence = MIN_TRACKING_CONFIDENCE
        #   num_poses = 1
        #   output_segmentation_masks = False
        # Create self._pose = PoseLandmarker.create_from_options(options)
        ...

        # ── Hand Landmarker ──
        # Create HandLandmarkerOptions:
        #   base_options = BaseOptions(model_asset_path=hand_model_path)
        #   running_mode = RunningMode.VIDEO
        #   num_hands = 1
        #   min_hand_detection_confidence = 0.5
        #   min_tracking_confidence = 0.5
        # Create self._hands = HandLandmarker.create_from_options(options)
        ...

        self._last_timestamp_ms = -1

    def process(self, bgr_frame, timestamp_ms):
        """
        Run pose and hand estimation on a BGR frame.

        Args:
            bgr_frame: np.ndarray, shape (H, W, 3), dtype uint8, BGR color order
            timestamp_ms: int, monotonically increasing frame timestamp

        Returns:
            PoseResult with landmarks or None fields if detection failed
        """
        # IMPORTANT: Timestamps must be strictly increasing for VIDEO mode.
        # If timestamp_ms <= self._last_timestamp_ms, increment by 1.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # Convert BGR → RGB (MediaPipe expects RGB)
        # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Create mp.Image from numpy array:
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Run Pose ──
        # pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        #
        # Extract landmarks:
        #   If pose_result.pose_world_landmarks is not empty:
        #     world_lms = pose_result.pose_world_landmarks[0]  # first (only) person
        #     norm_lms = pose_result.pose_landmarks[0]
        #   Else: world_lms = None, norm_lms = None
        ...

        # ── Run Hands ──
        # hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        #
        # Extract landmarks:
        #   If hand_result.hand_landmarks is not empty:
        #     hand_lms = hand_result.hand_landmarks[0]     # first hand
        #     hand_world_lms = hand_result.hand_world_landmarks[0]
        #   Else: hand_lms = None, hand_world_lms = None
        ...

        # Return PoseResult(
        #     pose_world_landmarks=world_lms,
        #     pose_landmarks=norm_lms,
        #     hand_landmarks=hand_lms,
        #     hand_world_landmarks=hand_world_lms,
        #     timestamp_ms=timestamp_ms,
        # )
        ...

    def close(self):
        """Release MediaPipe resources."""
        # self._pose.close()
        # self._hands.close()
        ...
```

### Skeleton drawing utility

Damian and Jaden don't need this, but it's useful for debugging and for the final visualization. Include it in `pose.py`:

```python
def draw_landmarks_on_frame(bgr_frame, pose_result):
    """
    Draw pose skeleton overlay on a BGR frame (in-place).

    Uses the NORMALIZED landmarks (pose_landmarks), not world landmarks,
    because we need pixel coordinates for drawing.

    Args:
        bgr_frame: np.ndarray to draw on (modified in-place)
        pose_result: PoseResult from PoseTracker.process()

    Returns:
        bgr_frame with skeleton drawn on it
    """
    if pose_result.pose_landmarks is None:
        return bgr_frame

    h, w = bgr_frame.shape[:2]

    # Define the connections to draw (pairs of landmark indices)
    # Right arm: shoulder(12)-elbow(14)-wrist(16)
    # Left arm: shoulder(11)-elbow(13)-wrist(15)
    # Torso: shoulder(11)-shoulder(12), hip(23)-hip(24),
    #         shoulder(11)-hip(23), shoulder(12)-hip(24)
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (16, 20), (16, 18), (15, 19), (15, 17),
    ]

    landmarks = pose_result.pose_landmarks

    # Draw connections as lines
    for i, j in CONNECTIONS:
        lm_i = landmarks[i]
        lm_j = landmarks[j]
        # Only draw if both landmarks have decent visibility
        if lm_i.visibility > 0.5 and lm_j.visibility > 0.5:
            pt1 = (int(lm_i.x * w), int(lm_i.y * h))
            pt2 = (int(lm_j.x * w), int(lm_j.y * h))
            cv2.line(bgr_frame, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark dots
    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 0, 255) if idx in [12, 14, 16] else (0, 255, 0)  # red for right arm
            cv2.circle(bgr_frame, (cx, cy), 4, color, -1)

    return bgr_frame
```

### Standalone test

```python
if __name__ == "__main__":
    """Test pose tracker with webcam."""
    import time
    from camera import Camera

    cam = Camera()
    tracker = PoseTracker()

    frame_count = 0
    t0 = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        timestamp_ms = int((time.time() - t0) * 1000)
        result = tracker.process(frame, timestamp_ms)

        # Draw skeleton
        draw_landmarks_on_frame(frame, result)

        # Print landmark info
        if result.pose_world_landmarks is not None:
            rw = result.pose_world_landmarks[16]  # right wrist
            print(f"Right wrist world: x={rw.x:.3f}, y={rw.y:.3f}, z={rw.z:.3f}")

        if result.hand_landmarks is not None:
            print(f"Hand detected: {len(result.hand_landmarks)} landmarks")

        # FPS
        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()
```

---

## Key Implementation Details

### 1. VIDEO vs LIVE_STREAM Mode

Use `RunningMode.VIDEO`, not `LIVE_STREAM`.

- `VIDEO` mode: synchronous, call `detect_for_video(image, timestamp_ms)`, get result immediately
- `LIVE_STREAM` mode: asynchronous, requires a callback, more complex

`VIDEO` mode is simpler and gives us direct control over the processing pipeline. The only requirement is that timestamps must be **strictly monotonically increasing** — handle this with the guard in `process()`.

### 2. World Landmarks vs Normalized Landmarks

MediaPipe returns two sets of landmarks:

| Type | Coordinates | Use For |
|------|-------------|---------|
| `pose_landmarks` | x,y normalized 0-1 by image size, z relative depth | Drawing on frame |
| `pose_world_landmarks` | x,y,z in meters, origin at hip midpoint | 3D angle computation |

**Damian needs `pose_world_landmarks`** for joint angle extraction (meters, real 3D).
**You need `pose_landmarks`** for drawing skeletons on the frame (pixel coordinates).
Provide both in `PoseResult`.

### 3. Hand Landmarker

The hand landmarker runs separately from the pose landmarker. Both use the same `mp.Image` and timestamp. Running both costs ~5-10ms extra per frame.

For the gripper, Damian needs:
- `hand_landmarks[4]` = thumb tip
- `hand_landmarks[8]` = index finger tip
- The distance between them maps to gripper open/close

### 4. Model File Paths

The `.task` files are NOT bundled with the pip package. They must be downloaded:

```bash
mkdir -p models
wget -O models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
wget -O models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

If the model file is missing, raise a clear `FileNotFoundError` with instructions.

### 5. Fallback: Legacy API

If the Tasks API gives you trouble, you can pin `mediapipe==0.10.21` and use the old API:

```python
# FALLBACK ONLY — if Tasks API doesn't work
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)
results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
landmarks = results.pose_landmarks.landmark  # list of 33
world_landmarks = results.pose_world_landmarks.landmark  # list of 33
```

But you'd need to convert the results into `PoseResult` with the same interface so Damian's code doesn't break.

---

## Acceptance Criteria

1. `python pose.py` opens webcam, draws skeleton overlay, prints wrist coordinates
2. `PoseResult` is populated correctly — `pose_world_landmarks` has 33 landmarks when a person is visible
3. `hand_landmarks` is populated when a hand is visible in frame
4. No crashes when no person is detected (all fields are `None`)
5. Timestamps are handled correctly (no "timestamp must be monotonically increasing" errors)
6. FPS stays above 20 with both pose and hand landmarkers running

---

## How Your Code Connects to Others

**Input from Griffin:**
```python
ok, frame = camera.read()  # BGR np.ndarray (720, 1280, 3) uint8
```

**Your processing:**
```python
result = tracker.process(frame, timestamp_ms)  # PoseResult
```

**Output to Damian:**
```python
# Damian uses result.pose_world_landmarks for joint angles
# Damian uses result.hand_landmarks for gripper control
action = mapper.compute(result)  # returns np.ndarray(6) or None
```

**Output to Jaden (visualization):**
```python
# Jaden calls draw_landmarks_on_frame for the display
draw_landmarks_on_frame(frame, result)
```
