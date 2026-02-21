"""
pose.py — MediaPipe Pose + Hand landmark extraction using the Tasks API.

Wraps PoseLandmarker and HandLandmarker into a single PoseTracker class.
Outputs a PoseResult dataclass consumed by mapping.py.

Requires model files:
  - models/pose_landmarker_full.task
  - models/hand_landmarker.task
"""

import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from config import (
    HAND_MODEL_PATH,
    MIN_POSE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_PATH,
)


@dataclass
class PoseResult:
    """Result from a single frame of pose + hand estimation."""
    pose_world_landmarks: list | None = None
    """33 world landmarks (x, y, z in meters, origin at hip midpoint). None if no pose detected."""

    pose_landmarks: list | None = None
    """33 normalized landmarks (x, y in 0-1, z relative depth). None if no pose detected."""

    pose_visibility: list[float] | None = None
    """33 visibility scores (0.0-1.0). None if no pose detected."""

    hand_landmarks: list | None = None
    """21 hand landmarks (normalized x, y, z). None if no hand detected."""

    hand_world_landmarks: list | None = None
    """21 hand world landmarks (meters). None if no hand detected."""

    timestamp_ms: int = 0
    """Frame timestamp in milliseconds."""


class PoseTracker:
    """Wraps MediaPipe PoseLandmarker + HandLandmarker (Tasks API, VIDEO mode)."""

    def __init__(self, model_path: str = POSE_MODEL_PATH,
                 hand_model_path: str = HAND_MODEL_PATH) -> None:
        for path, name in [(model_path, "pose"), (hand_model_path, "hand")]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{name} model not found at '{path}'. Download it with:\n"
                    f"  mkdir -p models\n"
                    f"  wget -O {path} "
                    f"\"https://storage.googleapis.com/mediapipe-models/"
                    f"{'pose_landmarker/pose_landmarker_full' if name == 'pose' else 'hand_landmarker/hand_landmarker'}"
                    f"/float16/1/{os.path.basename(path)}\""
                )

        self._pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=MIN_POSE_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                output_segmentation_masks=False,
            )
        )

        self._hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self._last_timestamp_ms = -1

    def process(self, bgr_frame, timestamp_ms: int) -> PoseResult:
        """Run pose and hand estimation on a BGR frame."""
        # Timestamp monotonicity guard
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Pose detection
        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_world_landmarks:
            world_lms = pose_result.pose_world_landmarks[0]
            norm_lms = pose_result.pose_landmarks[0]
        else:
            world_lms = None
            norm_lms = None

        # Hand detection
        hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks:
            hand_lms = hand_result.hand_landmarks[0]
            hand_world_lms = hand_result.hand_world_landmarks[0]
        else:
            hand_lms = None
            hand_world_lms = None

        return PoseResult(
            pose_world_landmarks=world_lms,
            pose_landmarks=norm_lms,
            hand_landmarks=hand_lms,
            hand_world_landmarks=hand_world_lms,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._pose.close()
        self._hands.close()


def draw_landmarks_on_frame(bgr_frame, pose_result: PoseResult):
    """Draw pose skeleton overlay on a BGR frame (in-place). Uses normalized landmarks."""
    if pose_result.pose_landmarks is None:
        return bgr_frame

    h, w = bgr_frame.shape[:2]
    landmarks = pose_result.pose_landmarks

    # Upper body connections
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (16, 20), (16, 18), (15, 19), (15, 17),
    ]

    for i, j in CONNECTIONS:
        lm_i, lm_j = landmarks[i], landmarks[j]
        if lm_i.visibility > 0.5 and lm_j.visibility > 0.5:
            pt1 = (int(lm_i.x * w), int(lm_i.y * h))
            pt2 = (int(lm_j.x * w), int(lm_j.y * h))
            cv2.line(bgr_frame, pt1, pt2, (0, 255, 0), 2)

    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 0, 255) if idx in (12, 14, 16) else (0, 255, 0)
            cv2.circle(bgr_frame, (cx, cy), 4, color, -1)

    return bgr_frame


if __name__ == "__main__":
    import time

    from camera import Camera

    cam = Camera()
    tracker = PoseTracker()
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - t0) * 1000)
            result = tracker.process(frame, timestamp_ms)

            draw_landmarks_on_frame(frame, result)

            if result.pose_world_landmarks is not None:
                rw = result.pose_world_landmarks[16]
                print(f"Right wrist world: x={rw.x:.3f}, y={rw.y:.3f}, z={rw.z:.3f}")

            if result.hand_landmarks is not None:
                print(f"Hand detected: {len(result.hand_landmarks)} landmarks")

            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Pose Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
