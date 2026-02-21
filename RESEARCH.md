# POV Camera → SO-ARM101 Robotic Arm Teleoperation

## Deep Research Report

**Date:** February 19, 2026  
**Purpose:** Weekend hackathon — build a system where a chest/head-mounted GoPro tracks human arm movements and maps them to SO-ARM101 robotic arms in real-time.

---

## Executive Summary

**The goal:** Wear a GoPro, have a computer track your arms from the first-person video feed, and make SO-ARM101 robot arms mimic your movements in real-time.

**Verdict: Feasible for a hackathon, with caveats.** The recommended approach uses **MediaPipe Pose** for real-time upper body tracking from the egocentric camera, combined with **geometric joint angle extraction** and direct **servo position commands** to the SO-ARM101 via the Feetech serial bus. This avoids complex inverse kinematics entirely.

**Key challenge:** Egocentric (first-person) views only see arms partially — they enter from the bottom/sides of frame. This is fundamentally harder than third-person tracking, but MediaPipe handles it reasonably well when arms are in the field of view (reaching forward).

**Expected performance:** 25-30 FPS tracking, ~50-80ms total latency (camera → servo movement).

---

## Architecture Diagram

```
┌─────────────┐     USB/RTMP      ┌──────────────────┐
│   GoPro     │ ──────────────→   │  Computer (GPU)  │
│ (chest/head)│   live video      │                  │
└─────────────┘                   │  ┌────────────┐  │
                                  │  │ MediaPipe  │  │
                                  │  │ Pose       │  │
                                  │  │ Estimation │  │
                                  │  └─────┬──────┘  │
                                  │        │         │
                                  │  ┌─────▼──────┐  │
                                  │  │ Joint Angle│  │
                                  │  │ Extraction │  │
                                  │  │ (geometry) │  │
                                  │  └─────┬──────┘  │
                                  │        │         │
                                  │  ┌─────▼──────┐  │
                                  │  │ Angle →    │  │
                                  │  │ Servo Map  │  │
                                  │  │ + Clamp    │  │
                                  │  └─────┬──────┘  │
                                  │        │         │
                                  └────────┼─────────┘
                                           │ USB Serial
                                  ┌────────▼─────────┐
                                  │  SO-ARM101       │
                                  │  (6 DOF, STS3215)│
                                  │  via Feetech bus │
                                  └──────────────────┘
```

---

## 1. SO-ARM101 / SO-ARM101 Specifications

### Hardware
| Spec | Detail |
|------|--------|
| **DOF** | 6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |
| **Servos** | Feetech STS3215 (serial bus servos) |
| **Voltage** | 7.4V (standard) or 12V (high-torque variant) |
| **Stall Torque** | 16.5 kg·cm @ 6V (7.4V version) |
| **Communication** | Half-duplex UART serial bus (daisy-chained) |
| **Controller** | Waveshare servo driver board (USB-C to computer) |
| **Feedback** | Position, speed, load, voltage, temperature (from each servo) |
| **Resolution** | 4096 steps per revolution (0.088° per step) |
| **Protocol** | Feetech SCS/STS serial protocol |

### Joint Configuration
1. **Joint 1 — Shoulder Pan** (base rotation, yaw) — ID 1
2. **Joint 2 — Shoulder Lift** (shoulder pitch) — ID 2
3. **Joint 3 — Elbow Flex** (elbow pitch) — ID 3
4. **Joint 4 — Wrist Flex** (wrist pitch) — ID 4
5. **Joint 5 — Wrist Roll** (wrist rotation) — ID 5
6. **Joint 6 — Gripper** (open/close) — ID 6

### Control Interface
- **USB-C** from computer to Waveshare motor control board
- **Serial protocol** via `feetech-servo-sdk` Python library (or via LeRobot's built-in support)
- Motors are **daisy-chained** on a single serial bus
- Each motor has a unique ID (1-6)
- Position commands: write target position (0-4095) to servo register

### Software: LeRobot (Hugging Face)
- **Repository:** https://github.com/huggingface/lerobot
- **Install:** `pip install lerobot`
- **SO-100 setup:** `lerobot-find-port`, `lerobot-setup-motors`
- **Teleoperation:** Built-in leader-follower teleoperation (physical leader arm)
- **Key API:**
  ```python
  from lerobot.robots.so100 import SO100Follower
  robot = SO100Follower(config=...)
  robot.connect()
  robot.send_action(action_dict)  # dict of joint positions
  ```

### Direct Servo Control (Simpler for Hackathon)
```python
# Using feetech-servo-sdk directly
from scservo_sdk import *

port = PortHandler('/dev/ttyACM0')
packet = PacketHandler(0)  # Protocol version
port.openPort()
port.setBaudRate(1000000)

# Write position to servo ID 1 (0-4095)
packet.write2ByteTxRx(port, servo_id=1, addr=42, value=2048)
```

---

## 2. Pose Estimation Approaches (Ranked for Hackathon)

### 🥇 Rank 1: MediaPipe Pose (RECOMMENDED)
- **Why:** Real-time (30+ FPS on CPU), no GPU needed, excellent Python API, 33 body landmarks including detailed arm joints
- **Landmarks for arms:** shoulder (11/12), elbow (13/14), wrist (15/16), plus hand landmarks
- **3D support:** Provides estimated 3D coordinates (x, y, z) — z is depth relative to hip
- **Egocentric performance:** Works when arms are visible in frame (reaching forward). Struggles when arms are at sides / behind camera
- **Install:** `pip install mediapipe`
- **Latency:** ~15-30ms per frame on modern CPU
- **Links:**
  - https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
  - https://github.com/google-ai-edge/mediapipe

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,  # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)  # or GoPro stream URL
while cap.isOpened():
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Right arm: shoulder=12, elbow=14, wrist=16
        # Left arm: shoulder=11, elbow=13, wrist=15
```

### 🥈 Rank 2: MediaPipe Holistic (Arms + Hands)
- Same as above but includes detailed hand landmarks (21 per hand)
- Useful if you want to control the gripper based on hand open/close
- Slightly heavier than Pose alone
- **Note:** MediaPipe Holistic is being deprecated in favor of combined Pose + Hand solutions

### 🥉 Rank 3: ViTPose / HRNet
- **ViTPose:** Vision Transformer-based, very accurate, but needs GPU
- **HRNet:** High-Resolution Net, also very accurate
- Both are overkill for hackathon and slower without GPU
- Papers:
  - ViTPose: https://arxiv.org/abs/2204.12484
  - HRNet: https://arxiv.org/abs/1902.09212

### Rank 4: OpenPose
- Classic multi-person pose estimation
- Slower than MediaPipe, needs GPU for real-time
- More complex setup
- https://github.com/CMU-Perceptual-Computing-Lab/openpose

### Not Recommended for Hackathon:
- **MMPose** — great but complex setup
- **AlphaPose** — multi-person focused, heavy
- **Custom egocentric models** — require training data, no time

---

## 3. Egocentric Vision: The Key Challenge

### The Problem
From a chest/head-mounted camera:
- Arms enter the frame from the **bottom or sides**
- Arms are **partially visible** most of the time
- When arms are at your sides or behind you, they're **invisible**
- **No full skeleton** visible — only the parts reaching into the camera's FOV
- **Depth is ambiguous** from monocular camera

### What Works
- **Arms reaching forward:** Best case. Both arms clearly visible, MediaPipe tracks well
- **Manipulation tasks:** When hands are in front of you working on something — this is the sweet spot
- **Chest mount vs head mount:** Chest mount gives more consistent arm visibility; head mount has wider FOV but arms may be out of view when looking away

### Mitigation Strategies
1. **Constrain the workspace:** Only map movements when arms are visible (confidence > threshold)
2. **Hold last known position:** When tracking is lost, keep servos at last valid position
3. **Use confidence scores:** MediaPipe provides per-landmark visibility scores — ignore low-confidence detections
4. **Smoothing filter:** Apply exponential moving average to prevent jitter
5. **Chest mount preferred:** More stable arm visibility than head mount

### Relevant Research
- **Ego4D** — Meta's massive egocentric video dataset. Includes hand/object interaction understanding
  - https://ego4d-data.org/
  - Paper: https://arxiv.org/abs/2110.07058
- **Ego-Exo4D** — Paired egocentric + exocentric video dataset
  - https://arxiv.org/abs/2311.18259
- **Assembly101** — Egocentric procedural activity dataset
  - https://arxiv.org/abs/2203.14712
- **EgoBody** — Egocentric body pose estimation
  - https://arxiv.org/abs/2112.07445
- **UnrealEgo** — Synthetic egocentric pose estimation
  - https://arxiv.org/abs/2208.01633

### Key Insight for Hackathon
Don't try to solve full egocentric pose estimation. Instead:
- Accept that you only track arms when they're in view
- Focus on the **forward-reaching workspace** where MediaPipe works reliably
- This naturally maps to the SO-ARM101's useful workspace anyway

---

## 4. Human-to-Robot Mapping (Joint Retargeting)

### Approach 1: Direct Joint Angle Mapping (RECOMMENDED)
Extract joint angles from MediaPipe landmarks using simple trigonometry, then map directly to servo positions.

```python
import numpy as np

def angle_between_vectors(v1, v2):
    """Compute angle between two 3D vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def extract_arm_angles(landmarks):
    """Extract arm joint angles from MediaPipe landmarks."""
    # Right arm landmarks
    shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
    
    # Shoulder angle (angle between torso and upper arm)
    torso_vec = hip - shoulder
    upper_arm_vec = elbow - shoulder
    shoulder_angle = angle_between_vectors(torso_vec, upper_arm_vec)
    
    # Elbow angle (angle between upper arm and forearm)
    forearm_vec = wrist - elbow
    elbow_angle = angle_between_vectors(-upper_arm_vec, forearm_vec)
    
    return shoulder_angle, elbow_angle

def angle_to_servo(angle_rad, min_angle, max_angle, min_servo, max_servo):
    """Map a joint angle to servo position (0-4095)."""
    ratio = (angle_rad - min_angle) / (max_angle - min_angle)
    ratio = np.clip(ratio, 0.0, 1.0)
    return int(min_servo + ratio * (max_servo - min_servo))
```

### Approach 2: Cartesian Position → Inverse Kinematics
- Extract wrist/hand position from MediaPipe (x, y, z in camera frame)
- Run IK solver to find servo angles that reach that position
- More complex, introduces IK solver latency
- **Not recommended for hackathon** — too many edge cases

### Approach 3: End-to-End Learned Mapping
- Train a neural network to map pose landmarks → servo positions
- Requires training data collection
- **Not feasible for a weekend**

### Mapping Considerations
- **Coordinate frame:** MediaPipe uses normalized coordinates (0-1 for x/y, z is relative depth). Need to transform to robot frame
- **Scale difference:** Human arm is ~60cm, SO-ARM101 is much smaller. Map angles, not positions
- **Joint limits:** SO-ARM101 servos have physical limits. Clamp all outputs
- **Smoothing:** Apply low-pass filter to prevent servo jitter
- **Shoulder pan:** Map horizontal arm position to base rotation
- **Shoulder lift:** Map vertical arm angle to shoulder servo
- **Elbow:** Direct angle mapping
- **Wrist:** Can map wrist angle from MediaPipe hand landmarks
- **Gripper:** Map hand open/close (distance between thumb and index finger tips)

### Smoothing Filter
```python
class ExponentialSmoothing:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
```

---

## 5. GoPro Video Streaming

### Option 1: GoPro as USB Webcam (RECOMMENDED)
- GoPro Hero 8+ supports **USB webcam mode**
- Connect via USB-C, appears as standard webcam
- Works with OpenCV `cv2.VideoCapture(0)`
- **Latency:** ~100-150ms (acceptable)
- **Resolution:** 1080p @ 30fps

### Option 2: GoPro WiFi Stream
- GoPro streams via RTMP or HLS over WiFi
- Higher latency (~200-500ms)
- Less reliable than USB
- Use: `cv2.VideoCapture("udp://@:8554")`

### Option 3: Any USB Webcam
- Simpler alternative: use a regular wide-angle USB webcam mounted on chest
- Lower latency than GoPro
- Logitech C920/C930e with wide-angle lens works well

### Option 4: Phone Camera (via IP Webcam app)
- Android: IP Webcam app → streams MJPEG/RTSP
- iPhone: Use as continuity camera or third-party app
- `cv2.VideoCapture("http://phone-ip:8080/video")`

---

## 6. Existing Projects & Repositories

### Directly Relevant
| Project | Description | Link |
|---------|-------------|------|
| **LeRobot** | HF framework for SO-ARM101, includes teleoperation | https://github.com/huggingface/lerobot |
| **SO-ARM101** | Official hardware repo (STL, BOM, assembly) | https://github.com/TheRobotStudio/SO-ARM101 |
| **GELLO** | Low-cost teleoperation framework (physical leader arm) | https://github.com/wuphilipp/gello_software |
| **feetech-servo-sdk** | Python SDK for STS3215 servos | `pip install feetech-servo-sdk` |

### Vision-Based Teleoperation
| Project | Description | Link |
|---------|-------------|------|
| **AnyTeleop** | Vision-based teleoperation using hand tracking | https://github.com/dexsuite/dex-retargeting |
| **Bunny-VisionPro** | Apple Vision Pro → robot arm teleoperation | https://github.com/Dingry/BunnyVisionPro |
| **dex-retargeting** | Human hand → robot hand retargeting library | https://github.com/dexsuite/dex-retargeting |
| **UMI** | Universal Manipulation Interface | https://github.com/real-stanford/universal_manipulation_interface |

### MediaPipe + Robot Arm Projects
| Project | Description | Link |
|---------|-------------|------|
| **mediapipe-robot-arm** | MediaPipe controlling a servo arm | Search GitHub for "mediapipe robot arm" |
| **hand-gesture-robot** | Various implementations exist | Multiple repos on GitHub |

### Key Papers
| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **AnyTeleop** | 2023 | Vision-based teleoperation, hand retargeting | https://arxiv.org/abs/2307.04577 |
| **GELLO** | 2023 | Low-cost teleoperation framework | https://arxiv.org/abs/2309.13037 |
| **ACE** | 2024 | Action Chunking with Egocentric observations | https://arxiv.org/abs/2408.11451 |
| **Bunny-VisionPro** | 2024 | VR-based bimanual teleoperation | https://arxiv.org/abs/2407.03162 |
| **Open-TeleVision** | 2024 | Immersive teleoperation | https://arxiv.org/abs/2407.01512 |
| **UMI** | 2024 | Universal manipulation interface | https://arxiv.org/abs/2402.10329 |
| **ViTPose** | 2022 | Vision Transformer pose estimation | https://arxiv.org/abs/2204.12484 |
| **Ego4D** | 2022 | Egocentric video understanding | https://arxiv.org/abs/2110.07058 |
| **EgoBody** | 2022 | Egocentric body pose | https://arxiv.org/abs/2112.07445 |
| **HRNet** | 2019 | High-resolution pose estimation | https://arxiv.org/abs/1902.09212 |

---

## 7. Feasibility Assessment

### ✅ What's Feasible
- **MediaPipe tracking at 30 FPS** — yes, even on CPU
- **GoPro USB webcam streaming** — straightforward
- **Direct servo control** — simple serial commands
- **Basic joint angle mapping** — geometry, no ML needed
- **Single arm control** — definitely achievable in a weekend
- **Both arms** — achievable if single arm works

### ⚠️ Challenges
| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Arms leaving FOV | High | Hold last position, confidence threshold |
| Depth ambiguity (monocular) | Medium | Use MediaPipe's z-estimate, map angles not positions |
| Servo jitter | Medium | Exponential smoothing filter |
| Coordinate frame mapping | Medium | Careful calibration, manual offset tuning |
| GoPro latency | Low-Medium | Use USB mode, not WiFi |
| Joint limit violations | Low | Clamp servo values to safe range |

### ❌ What's NOT Feasible in a Weekend
- Full 6-DOF arm tracking from egocentric view (wrist roll is very hard)
- Training custom egocentric pose models
- Handling heavy occlusion gracefully
- Sub-30ms latency
- Controlling fingers/gripper precisely from monocular video

### Expected Latency Budget
| Stage | Latency |
|-------|---------|
| GoPro USB capture | ~30-50ms |
| MediaPipe inference | ~15-30ms |
| Angle calculation | ~1ms |
| Serial communication | ~5-10ms |
| Servo response | ~10-20ms |
| **Total** | **~60-110ms** |

---

## 8. Recommended Hackathon Implementation Plan

### Day 1 (Saturday): Foundation
**Morning (4h):**
1. Set up GoPro USB webcam streaming → OpenCV capture
2. Get MediaPipe Pose running on the video feed
3. Visualize landmarks on screen, confirm arm tracking works
4. Test with arms in different positions, note where tracking fails

**Afternoon (4h):**
4. Connect to SO-ARM101 via USB serial
5. Write basic servo control: move each joint independently
6. Calibrate: find servo min/max positions for each joint
7. Create joint angle → servo position mapping table

### Day 2 (Sunday): Integration
**Morning (4h):**
8. Implement joint angle extraction from MediaPipe landmarks
9. Map extracted angles to servo commands
10. Add smoothing filter
11. First end-to-end test: move arm → robot moves

**Afternoon (4h):**
12. Tune mapping parameters (scaling, offsets, limits)
13. Add confidence-based filtering (ignore bad detections)
14. Add second arm if time permits
15. Demo polish: visualization overlay, recording

### MVP Features (Must Have)
- [ ] GoPro → computer video feed
- [ ] MediaPipe arm landmark detection
- [ ] 3 joint control (shoulder pan, shoulder lift, elbow)
- [ ] Real-time servo movement
- [ ] Basic smoothing

### Stretch Goals
- [ ] All 5 joints (+ wrist flex, wrist roll)
- [ ] Gripper control via hand open/close detection
- [ ] Both arms simultaneously
- [ ] Web dashboard showing tracking + robot state
- [ ] Recording/playback of movements

### Minimal Code Structure
```
pov-arm-teleop/
├── main.py              # Main loop: capture → track → control
├── camera.py            # GoPro/webcam capture
├── tracker.py           # MediaPipe pose estimation wrapper
├── mapper.py            # Joint angle extraction + mapping
├── servo_controller.py  # SO-ARM101 serial communication
├── smoother.py          # Signal smoothing utilities
├── config.py            # Joint limits, servo IDs, calibration
├── calibrate.py         # Interactive calibration tool
└── requirements.txt     # mediapipe, opencv-python, pyserial, numpy
```

### requirements.txt
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
pyserial>=3.5
feetech-servo-sdk>=1.0.0
```

---

## 9. Quick-Start Skeleton Code

```python
#!/usr/bin/env python3
"""POV Camera → SO-ARM101 Teleoperation - Main Loop"""

import cv2
import mediapipe as mp
import numpy as np
from scservo_sdk import *

# === CONFIG ===
CAMERA_INDEX = 0  # or GoPro stream URL
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
SMOOTHING_ALPHA = 0.3

# Servo IDs
SHOULDER_PAN = 1
SHOULDER_LIFT = 2
ELBOW_FLEX = 3
WRIST_FLEX = 4
WRIST_ROLL = 5
GRIPPER = 6

# === SERVO SETUP ===
port = PortHandler(SERIAL_PORT)
packet = PacketHandler(0)
port.openPort()
port.setBaudRate(BAUDRATE)

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === SMOOTHING ===
class Smoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.val = None
    def update(self, v):
        self.val = v if self.val is None else self.alpha * v + (1 - self.alpha) * self.val
        return self.val

smoothers = {joint: Smoother(SMOOTHING_ALPHA) for joint in [SHOULDER_PAN, SHOULDER_LIFT, ELBOW_FLEX]}

def write_servo(servo_id, position):
    position = max(0, min(4095, int(position)))
    packet.write2ByteTxRx(port, servo_id, 42, position)

def angle_3pts(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1))

def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z])

# === MAIN LOOP ===
cap = cv2.VideoCapture(CAMERA_INDEX)
print("Starting teleoperation... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Check visibility
        if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
            shoulder = landmark_to_np(lm[12])
            elbow = landmark_to_np(lm[14])
            wrist = landmark_to_np(lm[16])
            hip = landmark_to_np(lm[24])
            
            # Shoulder pan: horizontal angle of upper arm
            pan_angle = np.arctan2(elbow.x - shoulder.x, elbow.z - shoulder.z)
            pan_servo = int(2048 + pan_angle * 2048 / np.pi)
            
            # Shoulder lift: angle between torso and upper arm
            lift_angle = angle_3pts(hip, shoulder, elbow)
            lift_servo = int(np.interp(lift_angle, [0, np.pi], [1024, 3072]))
            
            # Elbow flex
            elbow_angle = angle_3pts(shoulder, elbow, wrist)
            elbow_servo = int(np.interp(elbow_angle, [0, np.pi], [1024, 3072]))
            
            # Apply smoothing and send
            write_servo(SHOULDER_PAN, smoothers[SHOULDER_PAN].update(pan_servo))
            write_servo(SHOULDER_LIFT, smoothers[SHOULDER_LIFT].update(lift_servo))
            write_servo(ELBOW_FLEX, smoothers[ELBOW_FLEX].update(elbow_servo))
        
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('POV Teleoperation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
port.closePort()
```

---

## 10. References

### Core Repositories
1. **LeRobot** — https://github.com/huggingface/lerobot
2. **SO-ARM101/101** — https://github.com/TheRobotStudio/SO-ARM101
3. **MediaPipe** — https://github.com/google-ai-edge/mediapipe
4. **GELLO** — https://github.com/wuphilipp/gello_software
5. **dex-retargeting** — https://github.com/dexsuite/dex-retargeting
6. **Bunny-VisionPro** — https://github.com/Dingry/BunnyVisionPro

### Key Papers
1. AnyTeleop (2023) — https://arxiv.org/abs/2307.04577
2. GELLO (2023) — https://arxiv.org/abs/2309.13037
3. ACE (2024) — https://arxiv.org/abs/2408.11451
4. Open-TeleVision (2024) — https://arxiv.org/abs/2407.01512
5. UMI (2024) — https://arxiv.org/abs/2402.10329
6. Ego4D (2022) — https://arxiv.org/abs/2110.07058
7. ViTPose (2022) — https://arxiv.org/abs/2204.12484
8. HRNet (2019) — https://arxiv.org/abs/1902.09212

### LeRobot Documentation
- SO-100 Setup: https://huggingface.co/docs/lerobot/so100
- SO-101 Setup: https://huggingface.co/docs/lerobot/so101
- Hardware Integration: https://huggingface.co/docs/lerobot/integrate_hardware
