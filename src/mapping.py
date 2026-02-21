"""
mapping.py — Convert MediaPipe landmarks to SO-ARM101 joint actions.

Takes a PoseResult from pose.py and returns a 6D numpy array of joint angles
in radians, clamped to the SO-ARM101's actual joint limits.

Supports both right arm (JointMapper) and left arm (LeftJointMapper).
"""

import numpy as np

from config import (
    JOINT_LIMITS,
    JOINT_NAMES,
    LEFT_JOINT_LIMITS,
    LEFT_JOINT_NAMES,
    MIN_VISIBILITY,
    REQUIRED_LANDMARKS,
    REQUIRED_LANDMARKS_LEFT,
    LM_RIGHT_SHOULDER,
    LM_RIGHT_ELBOW,
    LM_RIGHT_WRIST,
    LM_RIGHT_HIP,
    LM_RIGHT_INDEX,
    LM_RIGHT_PINKY,
    LM_LEFT_SHOULDER,
    LM_LEFT_ELBOW,
    LM_LEFT_WRIST,
    LM_LEFT_HIP,
    LM_LEFT_INDEX,
    LM_LEFT_PINKY,
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


def _compute_wrist_roll(forearm_vec, index_pt, pinky_pt, negate: bool) -> float:
    """
    Compute wrist roll angle from forearm axis and hand lateral vector.

    Args:
        forearm_vec: wrist - elbow vector
        index_pt: index finger position
        pinky_pt: pinky finger position
        negate: True for right arm (third-person mirror), False for left arm

    Returns:
        Wrist roll angle in radians.
    """
    forearm_len = np.linalg.norm(forearm_vec)
    eps = 1e-8
    if forearm_len <= eps:
        return 0.0

    forearm_unit = forearm_vec / forearm_len
    hand_lateral = pinky_pt - index_pt
    hand_lateral = hand_lateral - np.dot(hand_lateral, forearm_unit) * forearm_unit
    lat_len = np.linalg.norm(hand_lateral)
    if lat_len <= eps:
        return 0.0

    hand_lateral = hand_lateral / lat_len
    gravity = np.array([0.0, 1.0, 0.0])  # MediaPipe y = downward
    gravity_perp = gravity - np.dot(gravity, forearm_unit) * forearm_unit
    grav_len = np.linalg.norm(gravity_perp)
    if grav_len <= eps:
        return 0.0

    gravity_perp = gravity_perp / grav_len
    cos_r = np.clip(np.dot(hand_lateral, gravity_perp), -1.0, 1.0)
    sin_r = np.dot(np.cross(gravity_perp, hand_lateral), forearm_unit)
    roll = np.arctan2(sin_r, cos_r)
    return float(-roll if negate else roll)


def _extract_arm_angles(world_landmarks) -> dict:
    """
    Extract 5 right arm joint angles from world landmarks.

    MediaPipe world coords are hip-centered (meters). Third-person camera:
    horizontal angles (pan, roll) are negated to mirror user motion.
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
    pinky_pt = vec(LM_RIGHT_PINKY)

    upper_arm = elbow - shoulder
    shoulder_pan = -np.arctan2(upper_arm[0], -upper_arm[2])

    shoulder_lift = _angle_3pts(hip, shoulder, elbow)
    elbow_flex = _angle_3pts(shoulder, elbow, wrist)
    wrist_flex = _angle_3pts(elbow, wrist, index_pt)

    forearm = wrist - elbow
    wrist_roll = _compute_wrist_roll(forearm, index_pt, pinky_pt, negate=True)

    return {
        "shoulder_pan": float(shoulder_pan),
        "shoulder_lift": float(shoulder_lift),
        "elbow_flex": float(elbow_flex),
        "wrist_flex": float(wrist_flex),
        "wrist_roll": float(wrist_roll),
    }


def _extract_left_arm_angles(world_landmarks) -> dict:
    """
    Extract 5 left arm joint angles from world landmarks.

    Both robots face the same direction (same quat), so the third-person
    mirror negation on shoulder_pan and wrist_roll is the SAME as the right arm.
    The only difference is which landmark indices we read.
    """
    n = len(world_landmarks)
    def vec(i: int) -> np.ndarray:
        if i < 0 or i >= n:
            raise IndexError(f"Landmark index {i} out of range [0, {n})")
        return _lm_to_vec(world_landmarks[i])

    shoulder = vec(LM_LEFT_SHOULDER)
    elbow = vec(LM_LEFT_ELBOW)
    wrist = vec(LM_LEFT_WRIST)
    hip = vec(LM_LEFT_HIP)
    index_pt = vec(LM_LEFT_INDEX)
    pinky_pt = vec(LM_LEFT_PINKY)

    upper_arm = elbow - shoulder
    # Same negation as right arm — both robots have identical orientation
    shoulder_pan = -np.arctan2(upper_arm[0], -upper_arm[2])

    shoulder_lift = _angle_3pts(hip, shoulder, elbow)
    elbow_flex = _angle_3pts(shoulder, elbow, wrist)
    wrist_flex = _angle_3pts(elbow, wrist, index_pt)

    forearm = wrist - elbow
    wrist_roll = _compute_wrist_roll(forearm, index_pt, pinky_pt, negate=True)

    return {
        "shoulder_pan": float(shoulder_pan),
        "shoulder_lift": float(shoulder_lift),
        "elbow_flex": float(elbow_flex),
        "wrist_flex": float(wrist_flex),
        "wrist_roll": float(wrist_roll),
    }


def _extract_gripper(hand_landmarks, limits_key="gripper") -> float:
    """
    Extract gripper openness from hand landmarks (21 landmarks, normalized).
    Thumb tip = 4, index tip = 8. Maps distance to gripper joint range.
    """
    grip_min, grip_max = JOINT_LIMITS[limits_key] if limits_key in JOINT_LIMITS else LEFT_JOINT_LIMITS[limits_key]
    if hand_landmarks is None or len(hand_landmarks) < 9:
        return grip_min + 0.3 * (grip_max - grip_min)

    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    dist = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )
    normalized = np.clip((dist - 0.02) / (0.15 - 0.02), 0.0, 1.0)
    return float(grip_min + normalized * (grip_max - grip_min))


def _human_to_robot(angles_dict: dict, gripper_value: float,
                    joint_names=None, joint_limits=None) -> np.ndarray:
    """
    Map extracted human joint angles to SO-ARM101 action space.

    Robot joint conventions (empirically verified):
      shoulder_pan:  +val -> arm rotates to +X (right in POV camera)
      shoulder_lift: +val -> arm tilts DOWN, -val -> arm tilts UP
      elbow_flex:    +val -> elbow bends DOWN
      wrist_flex:    +val -> wrist curls DOWN
      wrist_roll:    rotates gripper around forearm axis
    """
    if joint_names is None:
        joint_names = JOINT_NAMES
    if joint_limits is None:
        joint_limits = JOINT_LIMITS

    action = np.zeros(6, dtype=np.float64)
    action[0] = angles_dict["shoulder_pan"]
    action[1] = np.pi / 2 - angles_dict["shoulder_lift"]  # invert: small human angle (arm down) -> +val (robot down)
    action[2] = np.pi - angles_dict["elbow_flex"]          # straight arm (pi) -> 0, bent -> positive
    action[3] = np.pi - angles_dict["wrist_flex"]          # straight wrist (pi) -> 0, bent -> positive
    action[4] = angles_dict["wrist_roll"]
    action[5] = gripper_value

    for i, name in enumerate(joint_names):
        lo, hi = joint_limits[name]
        action[i] = np.clip(action[i], lo, hi)
    return action


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


class JointMapper:
    """
    Converts PoseResult -> 6D robot action array (right arm).

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

        hand_lms = getattr(pose_result, "right_hand_landmarks", None)
        # Backward compat: fall back to hand_landmarks
        if hand_lms is None:
            hand_lms = getattr(pose_result, "hand_landmarks", None)
        if hand_lms is not None:
            if hand_lms and isinstance(hand_lms[0], (list, tuple)):
                first_hand = hand_lms[0]
            else:
                first_hand = hand_lms
            gripper = _extract_gripper(first_hand, limits_key="gripper")
        else:
            gripper = self._default_gripper

        action = _human_to_robot(angles, gripper)
        return action


class LeftJointMapper:
    """
    Converts PoseResult -> 6D robot action array (left arm).

    Usage:
        mapper = LeftJointMapper()
        action = mapper.compute(pose_result)  # np.ndarray(6) or None
    """

    def __init__(self) -> None:
        grip_min, grip_max = LEFT_JOINT_LIMITS["left_gripper"]
        self._default_gripper = grip_min + 0.3 * (grip_max - grip_min)

    def compute(self, pose_result) -> np.ndarray | None:
        """
        Extract left arm joint angles from a PoseResult and return a 6D action array.

        Returns:
            np.ndarray of shape (6,) with joint angles in radians, or
            None if required landmarks are not visible.
        """
        if pose_result is None:
            return None
        if not _check_visibility(pose_result, REQUIRED_LANDMARKS_LEFT):
            return None

        world_landmarks = _get_world_landmarks(pose_result)
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        try:
            angles = _extract_left_arm_angles(world_landmarks)
        except (IndexError, KeyError, AttributeError):
            return None

        hand_lms = getattr(pose_result, "left_hand_landmarks", None)
        if hand_lms is not None:
            if hand_lms and isinstance(hand_lms[0], (list, tuple)):
                first_hand = hand_lms[0]
            else:
                first_hand = hand_lms
            gripper = _extract_gripper(first_hand, limits_key="left_gripper")
        else:
            gripper = self._default_gripper

        action = _human_to_robot(angles, gripper,
                                  joint_names=LEFT_JOINT_NAMES,
                                  joint_limits=LEFT_JOINT_LIMITS)
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
        def __init__(self, pose_landmarks, pose_world_landmarks=None,
                     right_hand_landmarks=None, left_hand_landmarks=None):
            self.pose_landmarks = pose_landmarks
            self.pose_world_landmarks = pose_world_landmarks or pose_landmarks
            self.pose_visibility = None
            self.right_hand_landmarks = right_hand_landmarks
            self.left_hand_landmarks = left_hand_landmarks

        @property
        def hand_landmarks(self):
            return self.right_hand_landmarks

    # Build 33 fake world landmarks; only indices we use need sensible values
    def make_fake_world_landmarks():
        landmarks = [FakeLandmark(0.0, 0.0, 0.0)] * 33
        # Right arm
        landmarks[LM_RIGHT_HIP] = FakeLandmark(0.0, -0.1, 0.0)
        landmarks[LM_RIGHT_SHOULDER] = FakeLandmark(0.0, 0.0, 0.0)
        landmarks[LM_RIGHT_ELBOW] = FakeLandmark(0.2, 0.0, -0.3)
        landmarks[LM_RIGHT_WRIST] = FakeLandmark(0.4, 0.0, -0.4)
        landmarks[LM_RIGHT_INDEX] = FakeLandmark(0.5, 0.0, -0.4)
        landmarks[LM_RIGHT_PINKY] = FakeLandmark(0.5, 0.05, -0.35)
        # Left arm
        landmarks[LM_LEFT_HIP] = FakeLandmark(0.0, -0.1, 0.0)
        landmarks[LM_LEFT_SHOULDER] = FakeLandmark(0.0, 0.0, 0.0)
        landmarks[LM_LEFT_ELBOW] = FakeLandmark(-0.2, 0.0, -0.3)
        landmarks[LM_LEFT_WRIST] = FakeLandmark(-0.4, 0.0, -0.4)
        landmarks[LM_LEFT_INDEX] = FakeLandmark(-0.5, 0.0, -0.4)
        landmarks[LM_LEFT_PINKY] = FakeLandmark(-0.5, 0.05, -0.35)
        return landmarks

    fake_pose = FakePoseResult(
        make_fake_world_landmarks(),
        pose_world_landmarks=make_fake_world_landmarks(),
        right_hand_landmarks=[FakeLandmark(0.1 * i, 0.1 * i, 0) for i in range(21)],
        left_hand_landmarks=[FakeLandmark(0.1 * i, 0.1 * i, 0) for i in range(21)],
    )

    # Test right arm
    mapper = JointMapper()
    action = mapper.compute(fake_pose)
    if action is not None:
        print("Right Action (rad):", np.round(action, 3))
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            assert lo <= action[i] <= hi, f"{name} out of range"
        print("All right joints within limits.")
    else:
        print("No right action (visibility or landmarks failed)")

    # Test left arm
    left_mapper = LeftJointMapper()
    left_action = left_mapper.compute(fake_pose)
    if left_action is not None:
        print("Left Action (rad):", np.round(left_action, 3))
        for i, name in enumerate(LEFT_JOINT_NAMES):
            lo, hi = LEFT_JOINT_LIMITS[name]
            assert lo <= left_action[i] <= hi, f"{name} out of range"
        print("All left joints within limits.")
    else:
        print("No left action (visibility or landmarks failed)")

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
        left_mapper = LeftJointMapper()
        t0 = time.time()

        while True:
            ok, frame = cam.read()
            if not ok:
                break
            timestamp_ms = int((time.time() - t0) * 1000)
            result = tracker.process(frame, timestamp_ms)
            action = mapper.compute(result)
            left_action = left_mapper.compute(result)

            y_offset = 30
            if action is not None:
                labels = [
                    f"R pan={action[0]:+.2f}",
                    f"lift={action[1]:+.2f}",
                    f"elbow={action[2]:+.2f}",
                    f"wrist={action[3]:+.2f}",
                    f"roll={action[4]:+.2f}",
                    f"grip={action[5]:+.2f}",
                ]
                info = "  ".join(labels)
                cv2.putText(frame, info, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                y_offset += 25

            if left_action is not None:
                labels = [
                    f"L pan={left_action[0]:+.2f}",
                    f"lift={left_action[1]:+.2f}",
                    f"elbow={left_action[2]:+.2f}",
                    f"wrist={left_action[3]:+.2f}",
                    f"roll={left_action[4]:+.2f}",
                    f"grip={left_action[5]:+.2f}",
                ]
                info = "  ".join(labels)
                cv2.putText(frame, info, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

            if action is None and left_action is None:
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
