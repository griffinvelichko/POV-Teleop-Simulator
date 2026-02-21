"""
pose.py — MediaPipe Pose + Hand landmark extraction using the Tasks API.

Wraps PoseLandmarker and HandLandmarker into a single PoseTracker class.
Outputs a PoseResult dataclass consumed by mapping.py.

Requires model files:
  - models/pose_landmarker_full.task
  - models/hand_landmarker.task
"""

from dataclasses import dataclass
import os

import cv2
import numpy as np

from config import (
    POSE_MODEL_PATH,
    HAND_MODEL_PATH,
    MIN_POSE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


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


def _ensure_models():
    """Raise FileNotFoundError with instructions if model files are missing."""
    if not os.path.isfile(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found: {POSE_MODEL_PATH}\n"
            "Download with:\n"
            "  mkdir -p models\n"
            '  wget -O models/pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"'
        )
    if not os.path.isfile(HAND_MODEL_PATH):
        raise FileNotFoundError(
            f"Hand model not found: {HAND_MODEL_PATH}\n"
            "Download with:\n"
            "  mkdir -p models\n"
            '  wget -O models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"'
        )


class PoseTracker:
    """
    Wraps MediaPipe PoseLandmarker + HandLandmarker.

    Usage:
        tracker = PoseTracker()
        result = tracker.process(bgr_frame, timestamp_ms)
        tracker.close()
    """

    def __init__(self, model_path=POSE_MODEL_PATH, hand_model_path=HAND_MODEL_PATH):
        _ensure_models()

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        # Pose Landmarker
        pose_base = BaseOptions(model_asset_path=model_path)
        pose_opts = PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=MIN_POSE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        self._pose = PoseLandmarker.create_from_options(pose_opts)

        # Hand Landmarker
        hand_base = BaseOptions(model_asset_path=hand_model_path)
        hand_opts = HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = HandLandmarker.create_from_options(hand_opts)

        self._last_timestamp_ms = -1
        self._mp_image_cls = None  # Lazy import to get mp.Image

    def _get_mp_image(self, rgb_frame: np.ndarray):
        """Create MediaPipe Image from RGB numpy array."""
        if self._mp_image_cls is None:
            import mediapipe as mp

            self._mp_image_cls = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
        return self._mp_image_cls(
            image_format=self._mp_format,
            data=rgb_frame,
        )

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> PoseResult:
        """
        Run pose and hand estimation on a BGR frame.

        Args:
            bgr_frame: np.ndarray, shape (H, W, 3), dtype uint8, BGR color order
            timestamp_ms: int, monotonically increasing frame timestamp

        Returns:
            PoseResult with landmarks or None fields if detection failed
        """
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._get_mp_image(rgb_frame)

        # Pose
        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_world_landmarks and pose_result.pose_landmarks:
            world_lms = pose_result.pose_world_landmarks[0]
            norm_lms = pose_result.pose_landmarks[0]
        else:
            world_lms = None
            norm_lms = None

        # Hands
        hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks and hand_result.hand_world_landmarks:
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
        """Release MediaPipe resources."""
        self._pose.close()
        self._hands.close()


def draw_landmarks_on_frame(bgr_frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
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
    CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (16, 20),
        (16, 18),
        (15, 19),
        (15, 17),
    ]
    landmarks = pose_result.pose_landmarks

    def vis(lm):
        return getattr(lm, "visibility", 1.0)

    for i, j in CONNECTIONS:
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        lm_i = landmarks[i]
        lm_j = landmarks[j]
        if vis(lm_i) > 0.5 and vis(lm_j) > 0.5:
            pt1 = (int(lm_i.x * w), int(lm_i.y * h))
            pt2 = (int(lm_j.x * w), int(lm_j.y * h))
            cv2.line(bgr_frame, pt1, pt2, (0, 255, 0), 2)

    for idx, lm in enumerate(landmarks):
        if vis(lm) > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 0, 255) if idx in [12, 14, 16] else (0, 255, 0)
            cv2.circle(bgr_frame, (cx, cy), 4, color, -1)

    return bgr_frame


if __name__ == "__main__":
    """Test pose tracker with webcam."""
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
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )

            cv2.imshow("Pose Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
