# Damian's Plan — ML Engineer

**Role:** Joint angle extraction, human-to-robot mapping, smoothing filter
**Files to create:** `mapping.py`, `smoother.py`

---

## What You Own

You build the **mathematical core** — the module that transforms raw 3D body landmarks into a 6-element robot action array. This is pure geometry and signal processing: no ML frameworks, no camera code, no simulator code. Your input is Torrin's `PoseResult`, your output is a `np.ndarray(6)` that Jaden feeds directly into the simulator.

```
Torrin's PoseResult → YOUR mapping.py + smoother.py → 6D action array → Jaden's sim.py
```

---

## File 1: `smoother.py`

**Purpose:** Exponential Moving Average (EMA) filter to eliminate jitter in joint commands.

### What to implement

```python
"""
smoother.py — Signal smoothing filters for stable joint control.

The ExponentialSmoother applies an EMA to each joint independently.
Tuning alpha:
  - 0.1 = very smooth, sluggish (good for demos)
  - 0.3 = balanced (default)
  - 0.6 = responsive, some jitter
  - 1.0 = no smoothing (raw)
"""

import numpy as np
from config import SMOOTHING_ALPHA, DEADBAND_THRESHOLD


class Smoother:
    """
    Exponential Moving Average filter with optional deadband.

    Usage:
        s = Smoother(alpha=0.3)
        smoothed = s.update(raw_values)  # np.ndarray → np.ndarray
        s.reset()
    """

    def __init__(self, alpha=SMOOTHING_ALPHA, num_joints=6):
        self.alpha = alpha
        self.num_joints = num_joints
        self.state = None
        self.deadband = DEADBAND_THRESHOLD

    def update(self, values):
        """
        Apply EMA smoothing to a joint value array.

        Args:
            values: np.ndarray of shape (num_joints,)

        Returns:
            np.ndarray of shape (num_joints,) — smoothed values
        """
        values = np.asarray(values, dtype=np.float64)

        if self.state is None:
            self.state = values.copy()
            return self.state.copy()

        # Apply EMA: new_state = alpha * new_value + (1 - alpha) * old_state
        raw_smoothed = self.alpha * values + (1.0 - self.alpha) * self.state

        # Deadband: if the change from current state is smaller than threshold,
        # don't update that joint (prevents micro-jitter)
        delta = np.abs(raw_smoothed - self.state)
        mask = delta > self.deadband
        self.state = np.where(mask, raw_smoothed, self.state)

        return self.state.copy()

    def reset(self):
        """Reset the filter state. Next update() will initialize from scratch."""
        self.state = None
```

### Standalone test for smoother

```python
if __name__ == "__main__":
    # Test: feed noisy data through smoother, verify output is smooth
    s = Smoother(alpha=0.3)

    # Simulate jittery joint readings
    np.random.seed(42)
    target = np.array([0.5, -1.0, 1.2, 0.3, 0.0, 0.5])

    print("Testing smoother with noisy data:")
    for i in range(20):
        noise = np.random.normal(0, 0.1, size=6)
        raw = target + noise
        smoothed = s.update(raw)
        print(f"  Step {i:2d}: raw[0]={raw[0]:+.3f}  smooth[0]={smoothed[0]:+.3f}")

    print(f"\nFinal state: {s.state}")
    print(f"Target was:  {target}")
    print(f"Error:       {np.abs(s.state - target).mean():.4f} (should be small)")
```

---

## File 2: `mapping.py`

**Purpose:** Extract 6 joint angles from MediaPipe landmarks and map them to the SO-ARM101 action space.

### Geometry Overview

You're computing joint angles from 3D landmark positions using dot products and `atan2`. The key landmarks (from `pose_world_landmarks`, in meters, hip-centered):

```
Right shoulder = index 12
Right elbow    = index 14
Right wrist    = index 16
Right hip      = index 24
Right index    = index 20  (fingertip, for wrist angle)
```

The 6 joints to extract:

| Joint | How to Compute | Notes |
|-------|---------------|-------|
| `shoulder_pan` | `atan2(upper_arm.x, -upper_arm.z)` | Horizontal rotation of upper arm in x/z plane |
| `shoulder_lift` | `angle(hip, shoulder, elbow)` | Angle between torso and upper arm |
| `elbow_flex` | `angle(shoulder, elbow, wrist)` | Angle at the elbow joint |
| `wrist_flex` | `angle(elbow, wrist, index)` | Angle at the wrist (needs fingertip landmark) |
| `wrist_roll` | `atan2(forearm.x, forearm.y)` | Forearm rotation estimate (noisy — consider locking to 0) |
| `gripper` | thumb-index distance from hand landmarks | 0.0 = closed, mapped to joint range |

### What to implement

```python
"""
mapping.py — Convert MediaPipe landmarks to SO-ARM101 joint actions.

Takes a PoseResult from pose.py and returns a 6D numpy array of joint angles
in radians, clamped to the SO-ARM101's actual joint limits.
"""

import numpy as np
from config import (
    JOINT_LIMITS, JOINT_NAMES, MIN_VISIBILITY, REQUIRED_LANDMARKS,
    LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST,
    LM_RIGHT_HIP, LM_RIGHT_INDEX, LM_RIGHT_THUMB,
)


def _lm_to_vec(landmark):
    """Convert a single MediaPipe landmark to a numpy 3-vector."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _angle_3pts(a, b, c):
    """
    Compute the angle at point b formed by rays b→a and b→c.

    Args:
        a, b, c: np.ndarray of shape (3,)

    Returns:
        float: angle in radians [0, pi]
    """
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _check_visibility(pose_result, indices):
    """
    Check if all required landmarks have sufficient visibility.

    Uses the NORMALIZED landmarks (pose_landmarks) for visibility scores,
    because world_landmarks don't carry per-landmark visibility in all versions.

    Args:
        pose_result: PoseResult from pose.py
        indices: list of landmark indices to check

    Returns:
        bool: True if all landmarks are visible enough
    """
    if pose_result.pose_landmarks is None:
        return False
    landmarks = pose_result.pose_landmarks
    return all(landmarks[i].visibility > MIN_VISIBILITY for i in indices)


def _extract_arm_angles(world_landmarks):
    """
    Extract 5 arm joint angles from world landmarks using geometric methods.

    Args:
        world_landmarks: list of 33 world landmarks (meters, hip-centered)

    Returns:
        dict: {joint_name: angle_in_radians} for 5 arm joints (no gripper)
    """
    # Get key 3D points
    shoulder = _lm_to_vec(world_landmarks[LM_RIGHT_SHOULDER])
    elbow = _lm_to_vec(world_landmarks[LM_RIGHT_ELBOW])
    wrist = _lm_to_vec(world_landmarks[LM_RIGHT_WRIST])
    hip = _lm_to_vec(world_landmarks[LM_RIGHT_HIP])
    index = _lm_to_vec(world_landmarks[LM_RIGHT_INDEX])

    # ── Joint 1: Shoulder Pan ──
    # Horizontal rotation of the upper arm in the x/z plane (top-down view).
    # atan2(x_component, -z_component) gives the pan angle.
    upper_arm = elbow - shoulder
    shoulder_pan = np.arctan2(upper_arm[0], -upper_arm[2])

    # ── Joint 2: Shoulder Lift ──
    # Angle between the torso (hip→shoulder) and the upper arm (shoulder→elbow).
    # When arm hangs straight down, this is ~0. When arm is horizontal, this is ~pi/2.
    shoulder_lift = _angle_3pts(hip, shoulder, elbow)

    # ── Joint 3: Elbow Flex ──
    # Angle at the elbow between upper arm and forearm.
    # Straight arm ≈ pi, bent arm < pi.
    elbow_flex = _angle_3pts(shoulder, elbow, wrist)

    # ── Joint 4: Wrist Flex ──
    # Angle at the wrist between forearm and hand direction.
    # Uses the index finger tip as the hand direction reference.
    wrist_flex = _angle_3pts(elbow, wrist, index)

    # ── Joint 5: Wrist Roll ──
    # Forearm rotation is difficult to extract from pose landmarks alone.
    # Approximate using the forearm vector's orientation in the x/y plane.
    # NOTE: This is the least reliable joint. If too noisy, lock to 0.0.
    forearm = wrist - elbow
    wrist_roll = np.arctan2(forearm[0], forearm[1])

    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }


def _extract_gripper(hand_landmarks):
    """
    Extract gripper openness from hand landmarks.

    Measures the distance between thumb tip (landmark 4) and index tip (landmark 8).
    Maps to a value within the gripper joint limits.

    Args:
        hand_landmarks: list of 21 hand landmarks (normalized x, y, z)

    Returns:
        float: gripper value in radians within JOINT_LIMITS["gripper"]
    """
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]

    dist = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2
        + (thumb_tip.y - index_tip.y) ** 2
    )

    # Map distance to gripper range:
    #   dist ≈ 0.02 → pinch (closed) → gripper min
    #   dist ≈ 0.15 → spread (open) → gripper max
    grip_min, grip_max = JOINT_LIMITS["gripper"]
    normalized = np.clip((dist - 0.02) / (0.15 - 0.02), 0.0, 1.0)
    return grip_min + normalized * (grip_max - grip_min)


def _human_to_robot(angles_dict, gripper_value):
    """
    Map extracted human joint angles to the SO-ARM101 action space.

    Applies offsets so that natural human poses correspond to reasonable robot poses,
    then clamps to the actual joint limits.

    Args:
        angles_dict: dict from _extract_arm_angles()
        gripper_value: float from _extract_gripper(), or default if no hand detected

    Returns:
        np.ndarray of shape (6,): [shoulder_pan, shoulder_lift, elbow_flex,
                                    wrist_flex, wrist_roll, gripper]
    """
    action = np.zeros(6, dtype=np.float64)

    # Direct mapping with offsets:
    # shoulder_pan: direct (already in correct frame)
    action[0] = angles_dict["shoulder_pan"]

    # shoulder_lift: offset so "arm at side" (angle≈0) maps to robot's down position
    # The raw angle is 0 when arm is along torso, pi when arm is straight up.
    # Subtract pi/2 so "arm horizontal forward" ≈ 0 in robot space.
    action[1] = angles_dict["shoulder_lift"] - np.pi / 2

    # elbow_flex: invert so "straight arm" (angle≈pi) maps to robot's extended position
    # Robot convention: 0 = straight, positive = bent
    action[2] = np.pi - angles_dict["elbow_flex"]

    # wrist_flex: offset so neutral wrist ≈ 0
    action[3] = angles_dict["wrist_flex"] - np.pi / 2

    # wrist_roll: direct
    action[4] = angles_dict["wrist_roll"]

    # gripper
    action[5] = gripper_value

    # Clamp each joint to its limits
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        action[i] = np.clip(action[i], lo, hi)

    return action


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

class JointMapper:
    """
    Converts PoseResult → 6D robot action array.

    Usage:
        mapper = JointMapper()
        action = mapper.compute(pose_result)  # np.ndarray(6) or None
    """

    def __init__(self):
        # Default gripper: slightly open
        grip_min, grip_max = JOINT_LIMITS["gripper"]
        self._default_gripper = grip_min + 0.3 * (grip_max - grip_min)

    def compute(self, pose_result):
        """
        Extract joint angles from a PoseResult and return a 6D action array.

        Args:
            pose_result: PoseResult from pose.py

        Returns:
            np.ndarray of shape (6,) with joint angles in radians, or
            None if required landmarks are not visible
        """
        # 1. Check visibility of required landmarks
        if not _check_visibility(pose_result, REQUIRED_LANDMARKS):
            return None

        # 2. Extract arm angles from world landmarks
        angles = _extract_arm_angles(pose_result.pose_world_landmarks)

        # 3. Extract gripper from hand landmarks (or use default)
        if pose_result.hand_landmarks is not None:
            gripper = _extract_gripper(pose_result.hand_landmarks)
        else:
            gripper = self._default_gripper

        # 4. Map human angles to robot action space
        action = _human_to_robot(angles, gripper)

        return action
```

### Standalone test for mapping

```python
if __name__ == "__main__":
    """Test joint mapper with live camera + pose tracker."""
    import time
    import cv2
    from camera import Camera
    from pose import PoseTracker

    cam = Camera()
    tracker = PoseTracker()
    mapper = JointMapper()

    t0 = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        timestamp_ms = int((time.time() - t0) * 1000)
        result = tracker.process(frame, timestamp_ms)
        action = mapper.compute(result)

        if action is not None:
            labels = [
                f"pan={action[0]:+.2f}",
                f"lift={action[1]:+.2f}",
                f"elbow={action[2]:+.2f}",
                f"wrist={action[3]:+.2f}",
                f"roll={action[4]:+.2f}",
                f"grip={action[5]:+.2f}",
            ]
            info = "  ".join(labels)
            print(f"Action: {info}")
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No arm detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Mapping Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()
```

---

## Key Implementation Details

### 1. World Landmarks vs Normalized Landmarks

**Use `pose_world_landmarks` for angle computation.** These are in meters, centered at the hip midpoint. The `pose_landmarks` (normalized 0-1) are distorted by image perspective and are only useful for drawing overlays.

However, use `pose_landmarks` for **visibility checking** — the visibility score is a per-landmark confidence from 0 to 1.

### 2. The Z-Coordinate Problem

MediaPipe's `z` value (depth) has ~10 degrees average angular error compared to motion capture systems. It's the noisiest coordinate.

**Mitigation:**
- The `Smoother` helps significantly — `z` jitter gets averaged out
- If `wrist_roll` is too noisy (it depends heavily on `z`), lock it to `0.0`
- For a backup, you could compute angles from x,y only (2D projection) — this works for shoulder_lift and elbow_flex when the arm is roughly in the frontal plane

### 3. The Mapping Offsets

The offsets in `_human_to_robot()` (`- np.pi/2`, `np.pi - angle`, etc.) are there because:
- Human "arm at rest" is not the same pose as robot "arm at rest"
- Human elbow angle convention is opposite to the robot's
- These offsets are starting values — they will need tuning during integration

**Expect to adjust these during the integration phase with Jaden.** The right way to tune:
1. Stand in a neutral pose → robot should be in a neutral pose
2. Extend arm forward → robot should extend forward
3. Bend elbow → robot should bend
4. If directions are flipped, negate the offset or change the sign

### 4. Gripper Distance Thresholds

The thumb-index distance thresholds (`0.02` for pinch, `0.15` for spread) are in normalized coordinates (0-1 of image width). These are approximate and may need tuning depending on camera distance.

### 5. Deadband in Smoother

The deadband prevents "micro-jitter" — tiny oscillations when the arm is stationary. Without it, noise in MediaPipe's landmark detection causes the robot to vibrate slightly even when you're holding still. The default `0.02` radians (~1.1 degrees) is a good starting point.

---

## Acceptance Criteria

1. `python smoother.py` runs the noise test and shows smoothed output converging to target
2. `python mapping.py` opens webcam, shows joint angles updating in real-time
3. When arm is not visible, `mapper.compute()` returns `None` (not garbage values)
4. All 6 action values are within their respective `JOINT_LIMITS`
5. Moving arm left/right changes `shoulder_pan`, up/down changes `shoulder_lift`, etc.
6. Hand pinch/spread changes `gripper` value
7. When arm is stationary, smoothed output doesn't jitter (deadband working)

---

## How Your Code Connects to Others

**Input from Torrin:**
```python
result = tracker.process(frame, timestamp_ms)  # PoseResult
```

**Your processing:**
```python
raw_action = mapper.compute(result)  # np.ndarray(6) or None
if raw_action is not None:
    action = smoother.update(raw_action)  # np.ndarray(6)
```

**Output to Jaden:**
```python
obs = sim.step(action)  # Jaden feeds your action array into gym-soarm
```
