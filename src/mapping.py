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


def _lm_to_vec(landmark) -> np.ndarray:
    """Convert a single MediaPipe landmark to a numpy 3-vector."""
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle at point b formed by rays b→a and b→c.

    Args:
        a, b, c: np.ndarray of shape (3,)

    Returns:
        float: angle in radians [0, pi]
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _check_visibility(pose_result, indices: list) -> bool:
    """
    Check if all required landmarks have sufficient visibility.

    Uses the NORMALIZED landmarks (pose_landmarks) for visibility scores.

    Args:
        pose_result: PoseResult from pose.py
        indices: list of landmark indices to check

    Returns:
        bool: True if all landmarks are visible enough
    """
    if pose_result.pose_landmarks is None:
        return False
    landmarks = pose_result.pose_landmarks
    for i in indices:
        if i >= len(landmarks):
            return False
        vis = getattr(landmarks[i], "visibility", 0.0)
        if vis <= MIN_VISIBILITY:
            return False
    return True


def _extract_arm_angles(world_landmarks: list) -> dict:
    """
    Extract 5 arm joint angles from world landmarks using geometric methods.

    Args:
        world_landmarks: list of 33 world landmarks (meters, hip-centered)

    Returns:
        dict: {joint_name: angle_in_radians} for 5 arm joints (no gripper)
    """
    shoulder = _lm_to_vec(world_landmarks[LM_RIGHT_SHOULDER])
    elbow = _lm_to_vec(world_landmarks[LM_RIGHT_ELBOW])
    wrist = _lm_to_vec(world_landmarks[LM_RIGHT_WRIST])
    hip = _lm_to_vec(world_landmarks[LM_RIGHT_HIP])
    index = _lm_to_vec(world_landmarks[LM_RIGHT_INDEX])

    upper_arm = elbow - shoulder
    shoulder_pan = np.arctan2(upper_arm[0], -upper_arm[2])

    shoulder_lift = _angle_3pts(hip, shoulder, elbow)
    elbow_flex = _angle_3pts(shoulder, elbow, wrist)
    wrist_flex = _angle_3pts(elbow, wrist, index)

    forearm = wrist - elbow
    wrist_roll = np.arctan2(forearm[0], forearm[1])

    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }


def _extract_gripper(hand_landmarks: list) -> float:
    """
    Extract gripper openness from hand landmarks.

    Measures the distance between thumb tip (landmark 4) and index tip (landmark 8).
    Maps to a value within the gripper joint limits.

    Args:
        hand_landmarks: list of 21 hand landmarks (normalized x, y, z)

    Returns:
        float: gripper value in radians within JOINT_LIMITS["gripper"]
    """
    if len(hand_landmarks) < 9:
        grip_min, grip_max = JOINT_LIMITS["gripper"]
        return grip_min + 0.3 * (grip_max - grip_min)
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    dist = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )
    grip_min, grip_max = JOINT_LIMITS["gripper"]
    normalized = np.clip((dist - 0.02) / (0.15 - 0.02), 0.0, 1.0)
    return grip_min + normalized * (grip_max - grip_min)


def _human_to_robot(angles_dict: dict, gripper_value: float) -> np.ndarray:
    """
    Map extracted human joint angles to the SO-ARM101 action space.

    Applies offsets and clamps to joint limits.

    Args:
        angles_dict: dict from _extract_arm_angles()
        gripper_value: float from _extract_gripper()

    Returns:
        np.ndarray of shape (6,): [shoulder_pan, shoulder_lift, elbow_flex,
                                    wrist_flex, wrist_roll, gripper]
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


class JointMapper:
    """
    Converts PoseResult → 6D robot action array.

    Usage:
        mapper = JointMapper()
        action = mapper.compute(pose_result)  # np.ndarray(6) or None
    """

    def __init__(self):
        grip_min, grip_max = JOINT_LIMITS["gripper"]
        self._default_gripper = grip_min + 0.3 * (grip_max - grip_min)

    def compute(self, pose_result) -> np.ndarray | None:
        """
        Extract joint angles from a PoseResult and return a 6D action array.

        Args:
            pose_result: PoseResult from pose.py

        Returns:
            np.ndarray of shape (6,) with joint angles in radians, or
            None if required landmarks are not visible
        """
        if not _check_visibility(pose_result, REQUIRED_LANDMARKS):
            return None
        if pose_result.pose_world_landmarks is None:
            return None

        angles = _extract_arm_angles(pose_result.pose_world_landmarks)

        if pose_result.hand_landmarks is not None:
            gripper = _extract_gripper(pose_result.hand_landmarks)
        else:
            gripper = self._default_gripper

        action = _human_to_robot(angles, gripper)
        return action


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
    try:
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
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(
                    frame, "No arm detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            cv2.imshow("Mapping Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
