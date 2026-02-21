"""
camera.py — Camera capture with optional GoPro Hero 11 Black fish-eye correction.

Usage:
    cam = Camera(device=0, undistort=False)
    ok, frame = cam.read()  # frame is BGR np.ndarray (H, W, 3) or None
    cam.release()
"""

import cv2
import numpy as np

from config import (
    CAMERA_DEVICE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    ENABLE_UNDISTORT,
    GOPRO_CAMERA_MATRIX,
    GOPRO_DIST_COEFFS,
)


class Camera:
    """OpenCV video capture wrapper with optional GoPro fish-eye undistortion."""

    def __init__(
        self,
        device=CAMERA_DEVICE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        undistort=ENABLE_UNDISTORT,
    ):
        self.device = device
        self._width = width
        self._height = height
        self._undistort = undistort

        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {device}. "
                f"Run detect_cameras() to find available indices."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._map1 = None
        self._map2 = None
        if undistort:
            self._init_undistort_maps(width, height)

    def _init_undistort_maps(self, w: int, h: int) -> None:
        """Pre-compute remap tables for fast per-frame undistortion."""
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            GOPRO_CAMERA_MATRIX,
            GOPRO_DIST_COEFFS,
            (w, h),
            1.0,
            (w, h),
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            GOPRO_CAMERA_MATRIX,
            GOPRO_DIST_COEFFS,
            np.eye(3),
            new_matrix,
            (w, h),
            cv2.CV_16SC2,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read a single frame from the camera.

        Returns:
            tuple: (success: bool, frame: np.ndarray | None)
                   frame is BGR, shape (H, W, 3), dtype uint8
                   If undistort is enabled, frame is already corrected
        """
        ret, frame = self._cap.read()
        if not ret:
            return False, None

        if self._undistort and frame is not None:
            frame = cv2.remap(
                frame, self._map1, self._map2, cv2.INTER_LINEAR
            )

        return True, frame

    def release(self) -> None:
        """Release the camera device."""
        self._cap.release()


def detect_cameras(max_index: int = 5) -> list[int]:
    """
    Probe camera indices 0..max_index and return list of working indices.
    Useful for finding the right device when multiple cameras are connected.
    """
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


if __name__ == "__main__":
    import time

    cam = Camera()
    print(f"Camera opened: device={cam.device}")
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
