"""
camera.py — Camera capture for the POV teleop pipeline.

Usage:
    cam = Camera(device=0)
    ok, frame = cam.read()  # frame is BGR np.ndarray (H, W, 3) or None
    cam.release()
"""

import cv2

from config import (
    CAMERA_DEVICE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
)


class Camera:
    """OpenCV video capture wrapper."""

    def __init__(
        self,
        device=CAMERA_DEVICE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    ):
        self.device = device

        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {device}. "
                f"Run detect_cameras() to find available indices."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> tuple[bool, object]:
        """
        Read a single frame from the camera.

        Returns:
            tuple: (success: bool, frame: np.ndarray | None)
                   frame is BGR, shape (H, W, 3), dtype uint8
        """
        return self._cap.read()

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
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description="Test camera connection")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_DEVICE,
        help="Camera device index (default: 0). Run with --list to find indices.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available camera indices and exit",
    )
    args = parser.parse_args()

    available = detect_cameras()
    print(f"Available cameras: {available}")

    if args.list:
        print("Use --camera N to test (e.g. python src/camera.py --camera 1)")
        sys.exit(0)

    if args.camera not in available:
        print(f"\nCamera {args.camera} not found. Try: --camera 0, --camera 1, etc.")
        sys.exit(1)

    print(f"\nOpening camera {args.camera}")
    print("Press 'q' to quit\n")

    cam = Camera(device=args.camera)
    fps_counter = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Camera read failed.")
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
    finally:
        cam.release()
        cv2.destroyAllWindows()
