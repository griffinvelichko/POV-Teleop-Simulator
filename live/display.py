"""
display.py -- Live teleop display with dual-arm robot joint dashboard.

Left panel: camera feed with skeleton overlay (flipped).
Right panel: dual-arm joint bar graphs showing commanded vs actual positions.
Bottom: numeric dashboard with FPS, connection status, and controls hint.
"""

import cv2
import numpy as np

JOINT_LABELS = ["pan", "lift", "elbow", "wrist", "roll", "grip"]


class LiveDisplay:
    """
    Display for live robot teleoperation.

    Usage:
        disp = LiveDisplay()
        frame = disp.render(camera_frame, right_cmd, right_actual, ...)
        key = disp.show(frame)   # returns key code (0 if none)
        disp.close()
    """

    def __init__(self, width=1280, height=550):
        self.width = width
        self.height = height
        self._dashboard_height = 70
        self._panel_height = height - self._dashboard_height
        self._panel_width = width // 2

    def _draw_arm_bars(
        self,
        panel: np.ndarray,
        commanded_rad: np.ndarray | None,
        actual_rad: np.ndarray | None,
        y_start: int,
        label: str,
        cmd_color: tuple,
        act_color: tuple = (0, 200, 0),
    ) -> None:
        """Draw horizontal bar graphs for one arm's 6 joints."""
        w = self._panel_width
        bar_height = 14
        spacing = 6
        margin_left = 15
        label_width = 50
        bar_x = margin_left + label_width
        bar_max_width = w - bar_x - 70  # room for value text

        # Arm label
        cv2.putText(
            panel, label, (margin_left, y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cmd_color, 1,
        )

        for i, jname in enumerate(JOINT_LABELS):
            y = y_start + i * (bar_height + spacing)

            # Joint name
            cv2.putText(
                panel, jname, (margin_left, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1,
            )

            # Background
            cv2.rectangle(
                panel, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                (50, 50, 50), -1,
            )

            # Center line (zero position)
            center_x = bar_x + bar_max_width // 2
            cv2.line(panel, (center_x, y), (center_x, y + bar_height), (80, 80, 80), 1)

            def val_to_x(v: float) -> int:
                # Map [-pi, pi] to [0, bar_max_width]
                normalized = (v + np.pi) / (2 * np.pi)
                return bar_x + int(np.clip(normalized, 0.0, 1.0) * bar_max_width)

            # Commanded bar (top half)
            if commanded_rad is not None and i < len(commanded_rad):
                cx = val_to_x(commanded_rad[i])
                x1, x2 = min(center_x, cx), max(center_x, cx)
                cv2.rectangle(panel, (x1, y + 1), (x2, y + bar_height // 2), cmd_color, -1)
                val_str = f"{np.degrees(commanded_rad[i]):+.0f}"
                cv2.putText(
                    panel, val_str, (bar_x + bar_max_width + 4, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, cmd_color, 1,
                )

            # Actual bar (bottom half)
            if actual_rad is not None and i < len(actual_rad):
                ax = val_to_x(actual_rad[i])
                x1, x2 = min(center_x, ax), max(center_x, ax)
                cv2.rectangle(
                    panel, (x1, y + bar_height // 2), (x2, y + bar_height - 1),
                    act_color, -1,
                )

    def render(
        self,
        camera_frame: np.ndarray,
        right_cmd: np.ndarray | None = None,
        right_actual: np.ndarray | None = None,
        left_cmd: np.ndarray | None = None,
        left_actual: np.ndarray | None = None,
        fps: float = 0.0,
        right_connected: bool = False,
        left_connected: bool = False,
        frozen: bool = False,
    ) -> np.ndarray:
        """Build the composited display frame."""
        # Left panel: camera (flipped)
        left_panel = cv2.resize(
            cv2.flip(camera_frame, 1),
            (self._panel_width, self._panel_height),
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.putText(
            left_panel, "CAMERA", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # Right panel: joint dashboard
        right_panel = np.zeros(
            (self._panel_height, self._panel_width, 3), dtype=np.uint8,
        )
        right_panel[:] = (30, 30, 30)

        # Title
        cv2.putText(
            right_panel, "ROBOT JOINTS", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        # Connection indicators
        r_color = (0, 255, 0) if right_connected else (0, 0, 200)
        l_color = (0, 255, 0) if left_connected else (0, 0, 200)
        cv2.circle(right_panel, (self._panel_width - 60, 16), 6, r_color, -1)
        cv2.putText(
            right_panel, "R", (self._panel_width - 50, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, r_color, 1,
        )
        cv2.circle(right_panel, (self._panel_width - 30, 16), 6, l_color, -1)
        cv2.putText(
            right_panel, "L", (self._panel_width - 20, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, l_color, 1,
        )

        if frozen:
            cv2.putText(
                right_panel, "FROZEN", (self._panel_width // 2 - 40, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )

        # Right arm bars
        self._draw_arm_bars(
            right_panel, right_cmd, right_actual,
            y_start=50, label="RIGHT",
            cmd_color=(0, 140, 255),  # orange BGR
        )

        # Left arm bars
        self._draw_arm_bars(
            right_panel, left_cmd, left_actual,
            y_start=280, label="LEFT",
            cmd_color=(255, 140, 0),  # blue BGR
        )

        # Legend
        ly = self._panel_height - 15
        cv2.putText(
            right_panel, "CMD", (15, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 140, 255), 1,
        )
        cv2.putText(
            right_panel, "ACTUAL", (60, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1,
        )

        top = np.hstack([left_panel, right_panel])

        # Bottom dashboard
        dashboard = np.zeros((self._dashboard_height, self.width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)

        # Right arm numeric (row 1)
        spacing = self.width // 7
        cv2.putText(
            dashboard, "R:", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1,
        )
        for i, jname in enumerate(JOINT_LABELS):
            if right_cmd is not None and i < len(right_cmd):
                text = f"{jname}: {right_cmd[i]:+.2f}"
            else:
                text = f"{jname}: ---"
            cv2.putText(
                dashboard, text, (40 + i * spacing, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

        # Left arm numeric (row 2)
        cv2.putText(
            dashboard, "L:", (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1,
        )
        for i, jname in enumerate(JOINT_LABELS):
            if left_cmd is not None and i < len(left_cmd):
                text = f"{jname}: {left_cmd[i]:+.2f}"
            else:
                text = f"{jname}: ---"
            cv2.putText(
                dashboard, text, (40 + i * spacing, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

        # FPS
        cv2.putText(
            dashboard, f"FPS: {fps:.0f}",
            (self.width - 100, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

        # Controls hint
        cv2.putText(
            dashboard, "q=quit  e=e-stop  h=home  f=freeze  s=stow",
            (self.width - 400, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1,
        )

        return np.vstack([top, dashboard])

    def show(self, frame: np.ndarray) -> int:
        """
        Display frame. Returns key code pressed (0 if none).

        The caller handles key interpretation (q, e, h, f, etc.).
        """
        cv2.imshow("POV Teleop (LIVE)", frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()
