"""
main.py — POV Teleop main loop.

Orchestrates: Camera → PoseTracker → JointMapper → Smoother → SimController → Display

Usage (from project root):
    python -m src.main                    # default webcam
    python -m src.main --camera 1         # specific camera index
    python -m src.main --no-sim           # camera + pose only (no simulator)
    python -m src.main --alpha 0.15       # smoother demo
"""

import argparse
import os
import sys
import time

# When run as __main__, set cwd to project root (for models/) and path so "config" etc. resolve
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_root, "src")
    os.chdir(_root)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from config import (
    CAMERA_DEVICE,
    SMOOTHING_ALPHA,
    HOME_POSITION,
)
from camera import Camera
from pose import PoseTracker, draw_landmarks_on_frame
from mapping import JointMapper
from smoother import Smoother
from sim import SimController
from display import Display


def parse_args():
    parser = argparse.ArgumentParser(
        description="POV Teleop: Body-cam → SO-ARM101 simulator"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_DEVICE,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=SMOOTHING_ALPHA,
        help="Smoothing alpha (0.1=smooth, 0.3=balanced, 0.6=responsive)",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="Run camera + pose only, no simulator",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing camera...")
    camera = Camera(device=args.camera)

    print("Initializing pose tracker...")
    tracker = PoseTracker()

    print("Initializing joint mapper and smoother...")
    mapper = JointMapper()
    smoother = Smoother(alpha=args.alpha)

    sim = None
    if not args.no_sim:
        print("Initializing simulator...")
        sim = SimController()
        sim.reset()

    print("Initializing display...")
    display = Display()

    last_action = HOME_POSITION.copy()
    frame_count = 0
    t_start = time.time()
    fps = 0.0

    print("Pipeline ready. Move your right arm in front of the camera.")
    print("Press 'q' to quit.\n")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera read failed.")
                break

            timestamp_ms = int((time.time() - t_start) * 1000)
            pose_result = tracker.process(frame, timestamp_ms)
            raw_action = mapper.compute(pose_result)

            if raw_action is not None:
                action = smoother.update(raw_action)
                last_action = action
            else:
                action = last_action

            sim_frame = None
            if sim is not None and action is not None:
                sim.step(action)
                sim_frame = sim.get_sim_frame()

            draw_landmarks_on_frame(frame, pose_result)

            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed > 0:
                fps = frame_count / elapsed

            display_frame = display.render(frame, sim_frame, action, fps)
            if not display.show(display_frame):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("Shutting down...")
    camera.release()
    tracker.close()
    if sim is not None:
        sim.close()
    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
