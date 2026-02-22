"""
pipeline.py — Teleop pipeline running in a dedicated daemon thread.

Runs the full camera → pose → mapper → smoother → sim loop and pushes
JPEG-encoded frames + state to a janus sync queue for the FastAPI server.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import janus
import numpy as np

# Add src/ to path so config, camera, pose, etc. resolve
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
os.chdir(_root)
if _src not in sys.path:
    sys.path.insert(0, _src)

from camera import Camera
from config import HOME_POSITION, SMOOTHING_ALPHA
from mapping import JointMapper, LeftJointMapper
from pose import PoseTracker, draw_landmarks_on_frame
from sim import SimController
from smoother import Smoother

JPEG_QUALITY = 80


@dataclass
class FrameBundle:
    """One iteration's output from the pipeline thread."""
    sim_jpeg: bytes | None
    camera_jpeg: bytes | None
    right_action: list[float]
    left_action: list[float]
    fps: float


def _encode_jpeg(bgr_frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buf.tobytes()


def run_pipeline(
    queue: janus.Queue,
    stop_event: threading.Event,
    camera_device: int = 0,
    alpha: float = SMOOTHING_ALPHA,
) -> None:
    """
    Pipeline loop — runs in a daemon thread.

    Produces FrameBundle objects and pushes them to the sync side of a janus queue.
    Drops frames on QueueFull to prevent backpressure.
    """
    sync_q = queue.sync_q
    camera = None
    tracker = None
    sim = None

    try:
        print("[pipeline] Initializing camera...", flush=True)
        camera = Camera(device=camera_device)

        print("[pipeline] Initializing pose tracker...", flush=True)
        tracker = PoseTracker()

        print("[pipeline] Initializing mappers and smoothers...", flush=True)
        right_mapper = JointMapper()
        left_mapper = LeftJointMapper()
        right_smoother = Smoother(alpha=alpha)
        left_smoother = Smoother(alpha=alpha)

        print("[pipeline] Initializing simulator...", flush=True)
        sim = SimController()
        sim.reset()

        last_right_action = HOME_POSITION.copy()
        last_left_action = HOME_POSITION.copy()
        frame_count = 0
        t_start = time.time()
        fps = 0.0

        print("[pipeline] Running.", flush=True)

        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                print("[pipeline] Camera read failed.")
                break

            timestamp_ms = int((time.time() - t_start) * 1000)
            pose_result = tracker.process(frame, timestamp_ms)

            # Right arm
            raw_right = right_mapper.compute(pose_result)
            if raw_right is not None:
                right_action = right_smoother.update(raw_right)
                last_right_action = right_action
            else:
                right_action = last_right_action

            # Left arm
            raw_left = left_mapper.compute(pose_result)
            if raw_left is not None:
                left_action = left_smoother.update(raw_left)
                last_left_action = left_action
            else:
                left_action = last_left_action

            # Step simulator
            dual_action = np.concatenate([right_action, left_action])
            sim.step(dual_action)
            sim_bgr = sim.get_sim_frame()

            # Draw skeleton on camera frame, then flip for display
            draw_landmarks_on_frame(frame, pose_result)
            camera_display = cv2.flip(frame, 1)

            # Encode to JPEG
            sim_jpeg = _encode_jpeg(sim_bgr) if sim_bgr is not None else None
            camera_jpeg = _encode_jpeg(camera_display)

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed > 0:
                fps = frame_count / elapsed

            bundle = FrameBundle(
                sim_jpeg=sim_jpeg,
                camera_jpeg=camera_jpeg,
                right_action=right_action.tolist(),
                left_action=left_action.tolist(),
                fps=round(fps, 1),
            )

            # Non-blocking put — drop frame if queue full
            try:
                sync_q.put_nowait(bundle)
            except janus.SyncQueueFull:
                pass

    except Exception as e:
        import traceback
        print(f"[pipeline] Error: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("[pipeline] Shutting down...", flush=True)
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        if sim is not None:
            sim.close()
        print("[pipeline] Done.", flush=True)
