"""
display.py — Split-screen visualization with joint dashboard.

Composites the camera feed and simulator view side-by-side,
with a two-row status bar showing right and left arm joint angles and FPS.
"""

import cv2
import numpy as np

from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, WINDOW_NAME, JOINT_NAMES, LEFT_JOINT_NAMES


class Display:
    """
    Split-screen display builder.

    Usage:
        disp = Display()
        frame = disp.render(camera_frame, sim_frame, action, fps, left_action=left_action)
        keep_going = disp.show(frame)
        disp.close()
    """

    def __init__(self, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT):
        self.width = width
        self.height = height
        self._dashboard_height = 70
        self._panel_height = height - self._dashboard_height
        self._panel_width = width // 2

    def render(
        self,
        camera_frame: np.ndarray,
        sim_frame: np.ndarray | None = None,
        action: np.ndarray | None = None,
        fps: float = 0.0,
        left_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build the composited display frame.

        Args:
            camera_frame: BGR np.ndarray from camera (with skeleton overlay drawn)
            sim_frame: BGR np.ndarray from simulator, or None (shows placeholder)
            action: np.ndarray(6) current right arm action, or None
            fps: float, current pipeline FPS
            left_action: np.ndarray(6) current left arm action, or None

        Returns:
            np.ndarray: BGR composited frame ready for cv2.imshow
        """
        left_panel = cv2.resize(
            camera_frame, (self._panel_width, self._panel_height), interpolation=cv2.INTER_LINEAR
        )

        if sim_frame is not None:
            right_panel = cv2.resize(
                sim_frame, (self._panel_width, self._panel_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            right_panel = np.zeros((self._panel_height, self._panel_width, 3), dtype=np.uint8)
            right_panel[:] = (30, 30, 30)
            cv2.putText(
                right_panel,
                "Waiting for sim...",
                (self._panel_width // 4, self._panel_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 100, 100),
                2,
            )

        cv2.putText(
            left_panel, "CAMERA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            right_panel, "SIMULATOR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        top = np.hstack([left_panel, right_panel])

        dashboard = np.zeros((self._dashboard_height, self.width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)

        # Row 1: Right arm (orange label)
        spacing = self.width // 7
        right_color = (0, 140, 255)  # orange in BGR
        cv2.putText(dashboard, "R:", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1)
        for i, name in enumerate(JOINT_NAMES):
            short = name[:5] if len(name) > 5 else name
            if action is not None and i < len(action):
                text = f"{short}: {action[i]:+.2f}"
            else:
                text = f"{short}: —"
            x = 40 + i * spacing
            cv2.putText(
                dashboard, text, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )

        # Row 2: Left arm (blue label)
        left_color = (255, 140, 0)  # blue in BGR
        cv2.putText(dashboard, "L:", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1)
        for i, name in enumerate(LEFT_JOINT_NAMES):
            # Strip "left_" prefix for display
            short_name = name.replace("left_", "")
            short = short_name[:5] if len(short_name) > 5 else short_name
            if left_action is not None and i < len(left_action):
                text = f"{short}: {left_action[i]:+.2f}"
            else:
                text = f"{short}: —"
            x = 40 + i * spacing
            cv2.putText(
                dashboard, text, (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )

        # FPS in top-right of dashboard
        cv2.putText(
            dashboard,
            f"FPS: {fps:.0f}",
            (self.width - 100, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        frame = np.vstack([top, dashboard])
        return frame

    def show(self, frame: np.ndarray) -> bool:
        """
        Display the frame in a named window.

        Args:
            frame: BGR np.ndarray from render()

        Returns:
            bool: True to continue, False if user pressed 'q' or ESC
        """
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and key != 27

    def close(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """Test display with synthetic data."""
    disp = Display()

    for i in range(300):
        cam = np.zeros((480, 640, 3), dtype=np.uint8)
        cam[:, :, 1] = 100
        cv2.putText(
            cam, "Fake Camera Feed", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        sim = np.zeros((480, 640, 3), dtype=np.uint8)
        sim[:, :, 0] = 100
        cv2.putText(
            sim, "Fake Sim View", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        t = i / 30.0
        action = np.array(
            [
                np.sin(t) * 0.5,
                np.cos(t) * 0.3,
                np.sin(t * 2) * 0.8,
                np.cos(t * 0.5) * 0.2,
                0.0,
                0.5 + 0.3 * np.sin(t),
            ]
        )
        left_action = np.array(
            [
                -np.sin(t) * 0.5,
                np.cos(t) * 0.3,
                np.sin(t * 2) * 0.8,
                np.cos(t * 0.5) * 0.2,
                0.0,
                0.5 + 0.3 * np.sin(t),
            ]
        )

        frame = disp.render(cam, sim, action, fps=30.0, left_action=left_action)
        if not disp.show(frame):
            break

    disp.close()
