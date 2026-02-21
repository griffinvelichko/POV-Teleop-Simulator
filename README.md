# POV Teleop — Dual-Arm Teleoperation

Control two simulated SO-ARM101 robotic arms using a webcam and computer vision. MediaPipe tracks both of your arms and hands; two MuJoCo robots mirror your movements in real-time.

## How It Works

```
Webcam → MediaPipe Pose + Hands → Joint Angle Mapping → MuJoCo Sim → Display
```

1. **Camera** captures the user at 640x480
2. **MediaPipe** detects 33 body landmarks + 2-hand landmarks with handedness classification
3. **Mapping** converts shoulder/elbow/wrist angles to 6-DOF robot joint targets per arm
4. **Smoother** applies EMA filtering to reduce jitter
5. **Simulator** steps two SO-ARM101 robots with the 12-DOF action (right + left)
6. **Display** shows a split-screen: camera feed (left) and sim POV (right) with a joint dashboard

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download MediaPipe models
mkdir -p models
wget -O models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
wget -O models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Run
python -m src.main
```

## Usage

```bash
python -m src.main                # default webcam
python -m src.main --camera 1     # specific camera index
python -m src.main --no-sim       # camera + pose only (no simulator)
python -m src.main --alpha 0.15   # smoother tuning (0.1=smooth, 0.7=responsive)
```

Press **q** to quit.

## Project Structure

```
src/
  main.py       — Main loop: orchestrates the full pipeline
  config.py     — All tunable parameters (camera, models, joints, display)
  camera.py     — OpenCV webcam capture
  pose.py       — MediaPipe pose + 2-hand landmark extraction
  mapping.py    — Landmark angles → robot joint actions (JointMapper, LeftJointMapper)
  smoother.py   — EMA filter with deadband
  sim.py        — gym-soarm MuJoCo environment wrapper
  display.py    — Split-screen compositing with dual-arm joint dashboard
models/
  pose_landmarker_lite.task   — MediaPipe pose model
  hand_landmarker.task        — MediaPipe hand model
```

## Dual-Arm Architecture

- **Right robot** at world position `(0, 0.15, 0)` — controlled by user's right arm
- **Left robot** at `(-0.20, 0.15, 0)` — controlled by user's left arm
- Both robots are identical SO-ARM101 6-DOF arms (shoulder pan/lift, elbow flex, wrist flex/roll, gripper)
- MuJoCo scene: `qpos(12)` in teleop mode (right arm `[0:6]`, left arm `[6:12]`)
- Gripper controlled by thumb-index finger distance from hand landmarks

## Robot Joint Conventions

| Joint | +value | -value |
|-------|--------|--------|
| shoulder_pan | Arm to +X (right in POV) | Arm to -X (left in POV) |
| shoulder_lift | Arm tilts DOWN | Arm tilts UP |
| elbow_flex | Elbow bends DOWN | Elbow straightens |
| wrist_flex | Wrist curls DOWN | Wrist straightens |
| wrist_roll | Forearm supination | Forearm pronation |
| gripper | Open | Closed |

## Dependencies

- Python 3.12
- MediaPipe (pose + hand landmark detection)
- MuJoCo (physics simulation via dm_control)
- gym-soarm (SO-ARM101 Gymnasium environment)
- OpenCV (camera capture + display)
- PyTorch (required by gym-soarm)

## License

MIT
