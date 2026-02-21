"""
mapping.py — Convert MediaPipe landmarks to SO-ARM101 joint actions.

Takes a PoseResult from pose.py and returns a 6D numpy array of joint angles
in radians, clamped to the SO-ARM101's actual joint limits.
"""

import numpy as np

from config import (
    JOINT_LIMITS,
    JOINT_NAMES,
    MIN_VISIBILITY,
    REQUIRED_LANDMARKS,
    LM_RIGHT_SHOULDER,
    LM_RIGHT_ELBOW,
    LM_RIGHT_WRIST,
    LM_RIGHT_HIP,
    LM_RIGHT_INDEX,
)

# Avoid circular import; pose module provides PoseResult
try:
    from pose import PoseResult
except ImportError:
    PoseResult = None  # type: ignore[misc, assignment]


def _lm_to_vec(landmark) -> np.ndarray:
    """Convert a single MediaPipe landmark to a numpy 3-vector (x, y, z)."""
    return np.array(
        [float(landmark.x), float(landmark.y), float(landmark.z)],
        dtype=np.float64,
    )


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle at point b formed by rays b->a and b->c.

    Args:
        a, b, c: np.ndarray of shape (3,)

    Returns:
        Angle in radians in [0, pi].
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    eps = 1e-8
    if n1 <= eps or n2 <= eps:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _get_visibility(pose_result, index: int) -> float:
    """
    Get visibility score for one landmark. Uses pose_visibility if present,
    otherwise landmark.visibility if present, else 0.0.
    Handles pose_landmarks as flat list of 33 or list-of-lists (first pose).
    """
    if getattr(pose_result, "pose_visibility", None) is not None:
        vis = pose_result.pose_visibility
        if 0 <= index < len(vis):
            return float(vis[index])
    pl = getattr(pose_result, "pose_landmarks", None)
    if pl is None:
        return 0.0
    # List of poses: take first pose's landmarks
    landmarks = pl[0] if pl and isinstance(pl[0], (list, tuple)) else pl
    if 0 <= index < len(landmarks):
        lm = landmarks[index]
        if hasattr(lm, "visibility"):
            return float(lm.visibility)
    return 0.0


def _check_visibility(pose_result, indices: list[int]) -> bool:
    """
    Return True if all required landmarks have sufficient visibility.

    Uses pose_visibility when available; otherwise per-landmark visibility
    on pose_landmarks (normalized landmarks often carry visibility).
    """
    if pose_result.pose_landmarks is None and getattr(pose_result, "pose_visibility", None) is None:
        return False
    for i in indices:
        if _get_visibility(pose_result, i) <= MIN_VISIBILITY:
            return False
    return True


def _get_world_landmarks(pose_result):
    """
    Return 33 world landmarks (meters, hip-centered) for angle computation.
    Prefer pose_world_landmarks; fall back to pose_landmarks. Handles
    list-of-lists (one list per detected pose) by taking the first pose.
    """
    raw = None
    if getattr(pose_result, "pose_world_landmarks", None) is not None:
        raw = pose_result.pose_world_landmarks
    elif getattr(pose_result, "pose_landmarks", None) is not None:
        raw = pose_result.pose_landmarks
    if raw is None:
        return None
    # MediaPipe can return list of lists (one per pose); take first pose
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return raw if len(raw) >= 33 else None


def _extract_arm_angles(world_landmarks) -> dict:
    """
    Extract 5 arm joint angles from world landmarks.

    Args:
        world_landmarks: list of 33 world landmarks (meters, hip-centered).
                         Each element has .x, .y, .z.

    Returns:
        dict: {joint_name: angle_in_radians} for 5 arm joints (no gripper).
    """
    n = len(world_landmarks)
    def vec(i: int) -> np.ndarray:
        if i < 0 or i >= n:
            raise IndexError(f"Landmark index {i} out of range [0, {n})")
        return _lm_to_vec(world_landmarks[i])

    shoulder = vec(LM_RIGHT_SHOULDER)
    elbow = vec(LM_RIGHT_ELBOW)
    wrist = vec(LM_RIGHT_WRIST)
    hip = vec(LM_RIGHT_HIP)
    index_pt = vec(LM_RIGHT_INDEX)

    upper_arm = elbow - shoulder
    shoulder_pan = np.arctan2(upper_arm[0], -upper_arm[2])

    shoulder_lift = _angle_3pts(hip, shoulder, elbow)
    elbow_flex = _angle_3pts(shoulder, elbow, wrist)
    wrist_flex = _angle_3pts(elbow, wrist, index_pt)

    forearm = wrist - elbow
    wrist_roll = np.arctan2(forearm[0], forearm[1])

    return {
        "shoulder_pan": float(shoulder_pan),
        "shoulder_lift": float(shoulder_lift),
        "elbow_flex": float(elbow_flex),
        "wrist_flex": float(wrist_flex),
        "wrist_roll": float(wrist_roll),
    }


def _extract_gripper(hand_landmarks) -> float:
    """
    Extract gripper openness from hand landmarks (21 landmarks, normalized).
    Thumb tip = 4, index tip = 8. Maps distance to gripper joint range.
    """
    if hand_landmarks is None or len(hand_landmarks) < 9:
        grip_min, grip_max = JOINT_LIMITS["gripper"]
        return grip_min + 0.3 * (grip_max - grip_min)

    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    dist = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )
    grip_min, grip_max = JOINT_LIMITS["gripper"]
    normalized = np.clip((dist - 0.02) / (0.15 - 0.02), 0.0, 1.0)
    return float(grip_min + normalized * (grip_max - grip_min))


def _human_to_robot(angles_dict: dict, gripper_value: float) -> np.ndarray:
    """
    Map extracted human joint angles to SO-ARM101 action space.
    Applies offsets and clamps to joint limits.
    """
    action = np.zeros(6, dtype=np.float64)
    action[0] = angles_dict["shoulder_pan"]
    action[1] = angles_dict["shoulder_lift"] - np.pi / 2
    action[2] = np.pi - angles_dict["elbow_flex"]
    action[3] = angles_dict["wrist_flex"] - np.pi / 2
    action[4] = angles_dict["wrist_roll"]
    action[5] = gripper_value

    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        action[i] = np.clip(action[i], lo, hi)
    return action


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


class JointMapper:
    """
    Converts PoseResult -> 6D robot action array.

    Usage:
        mapper = JointMapper()
        action = mapper.compute(pose_result)  # np.ndarray(6) or None
    """

    def __init__(self) -> None:
        grip_min, grip_max = JOINT_LIMITS["gripper"]
        self._default_gripper = grip_min + 0.3 * (grip_max - grip_min)

    def compute(self, pose_result) -> np.ndarray | None:
        """
        Extract joint angles from a PoseResult and return a 6D action array.

        Args:
            pose_result: PoseResult from pose.py (or object with same attributes).

        Returns:
            np.ndarray of shape (6,) with joint angles in radians, or
            None if required landmarks are not visible.
        """
        if pose_result is None:
            return None
        if not _check_visibility(pose_result, REQUIRED_LANDMARKS):
            return None

        world_landmarks = _get_world_landmarks(pose_result)
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        try:
            angles = _extract_arm_angles(world_landmarks)
        except (IndexError, KeyError, AttributeError):
            return None

        hand_lms = getattr(pose_result, "hand_landmarks", None)
        if hand_lms is not None:
            # hand_landmarks may be list of hands (each hand = list of 21) or single hand
            if hand_lms and isinstance(hand_lms[0], (list, tuple)):
                first_hand = hand_lms[0]
            else:
                first_hand = hand_lms
            gripper = _extract_gripper(first_hand)
        else:
            gripper = self._default_gripper

        action = _human_to_robot(angles, gripper)
        return action


# ──────────────────────────────────────────────
# Standalone tests
# ──────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    # Test 1: Unit test with fake PoseResult (no camera/pose required)
    print("Test 1: Hardcoded landmark values")
    print("-" * 50)

    class FakeLandmark:
        def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
            self.x, self.y, self.z = x, y, z
            self.visibility = visibility

    class FakePoseResult:
        def __init__(self, pose_landmarks, pose_world_landmarks=None, hand_landmarks=None):
            self.pose_landmarks = pose_landmarks
            self.pose_world_landmarks = pose_world_landmarks or pose_landmarks
            self.pose_visibility = None
            self.hand_landmarks = hand_landmarks

    # Build 33 fake world landmarks; only indices we use need sensible values
    def make_fake_world_landmarks():
        landmarks = [FakeLandmark(0.0, 0.0, 0.0)] * 33
        landmarks[LM_RIGHT_HIP] = FakeLandmark(0.0, -0.1, 0.0)
        landmarks[LM_RIGHT_SHOULDER] = FakeLandmark(0.0, 0.0, 0.0)
        landmarks[LM_RIGHT_ELBOW] = FakeLandmark(0.2, 0.0, -0.3)
        landmarks[LM_RIGHT_WRIST] = FakeLandmark(0.4, 0.0, -0.4)
        landmarks[LM_RIGHT_INDEX] = FakeLandmark(0.5, 0.0, -0.4)
        return landmarks

    fake_pose = FakePoseResult(
        make_fake_world_landmarks(),
        pose_world_landmarks=make_fake_world_landmarks(),
        hand_landmarks=[FakeLandmark(0.1 * i, 0.1 * i, 0) for i in range(21)],
    )
    mapper = JointMapper()
    action = mapper.compute(fake_pose)
    if action is not None:
        print("Action (rad):", np.round(action, 3))
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            assert lo <= action[i] <= hi, f"{name} out of range"
        print("All joints within limits.")
    else:
        print("No action (visibility or landmarks failed)")

    # Test 2: Live camera + pose (optional)
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        print("\nTest 2: Live camera + pose (press 'q' to quit)")
        print("-" * 50)
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
                cv2.putText(
                    frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            else:
                cv2.putText(
                    frame, "No arm detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.imshow("Mapping Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
    else:
        print("\nRun with --live to test with camera and pose tracker.")
