# POV Arm Teleop — Implementation Plan

**Date:** February 20, 2026  
**Hackathon Time Budget:** 6 hours  
**Target:** Working demo of body-cam → pose estimation → simulated SO-ARM101 control

---

## 1. Executive Summary

**What we're building:** Strap a camera to your chest, move your arm, and watch a simulated SO-ARM101 robot arm mirror your movements in real-time. No VR headset, no physical robot, no special hardware — just a camera and computer vision.

**The wow factor:** First-person "Iron Man" style robot control. The operator's natural arm movements are captured from their own point of view and instantly translated to robot joint commands. The simulated arm moves as you move — reach forward, it reaches forward; bend your elbow, it bends its elbow.

**Tech stack:** GoPro/phone camera → OpenCV capture → MediaPipe Pose → geometric joint angle extraction → exponential smoothing → gym-soarm MuJoCo simulator

**Why it matters:** This is zero-hardware teleoperation. Every other approach requires either a physical leader arm ($200+), a VR headset ($500+), or specialized sensors. Ours needs a phone you already own.

---

## 2. Architecture Diagram

```
┌─────────────────┐         ┌──────────────────────────────────────────────┐
│  CAMERA SOURCE  │         │              PROCESSING PIPELINE              │
│                 │  USB/   │                                              │
│  GoPro Hero 8+  │  WiFi   │  ┌────────────┐   ┌─────────────────────┐   │
│  (webcam mode)  │────────→│  │  OpenCV     │   │  MediaPipe Pose     │   │
│                 │         │  │  Capture    │──→│  (model_complexity  │   │
│  ── OR ──       │         │  │  + Undist.  │   │   =1, 33 landmarks) │   │
│                 │         │  └────────────┘   └──────────┬──────────┘   │
│  iPhone/Android │         │                              │              │
│  (DroidCam/     │         │  ┌───────────────────────────▼───────────┐  │
│   Continuity)   │         │  │  Joint Angle Extraction (geometric)   │  │
│                 │         │  │  shoulder_pan ← atan2(elbow.x, .z)    │  │
│  ── OR ──       │         │  │  shoulder_lift ← angle(hip,shldr,elb) │  │
│                 │         │  │  elbow_flex ← angle(shldr,elb,wrist)  │  │
│  Laptop webcam  │         │  │  wrist_flex ← angle(elb,wrist,hand)   │  │
│  (fallback)     │         │  │  wrist_roll ← forearm rotation est.   │  │
│                 │         │  │  gripper ← thumb-index distance       │  │
└─────────────────┘         │  └───────────────────────────┬───────────┘  │
                            │                              │              │
                            │  ┌───────────────────────────▼───────────┐  │
                            │  │  Smoothing + Safety                    │  │
                            │  │  • Exponential Moving Average (α=0.3)  │  │
                            │  │  • Visibility confidence gating (>0.5) │  │
                            │  │  • Joint limit clamping                │  │
                            │  │  • Deadband (ignore tiny changes)      │  │
                            │  └───────────────────────────┬───────────┘  │
                            │                              │              │
                            └──────────────────────────────┼──────────────┘
                                                           │
                            ┌──────────────────────────────▼──────────────┐
                            │          SIMULATOR (gym-soarm)               │
                            │                                              │
                            │  MuJoCo SO-ARM101 model                     │
                            │  • Gymnasium env: SoArm-v0                  │
                            │  • 6DOF action space (joint angles in rad)  │
                            │  • Multi-camera rendering (overview + wrist)│
                            │  • OpenCV visualization window              │
                            │  • ~60 FPS physics stepping                 │
                            └──────────────────────────────────────────────┘
```

**Latency Budget (end-to-end):**
| Stage | Estimated |
|-------|-----------|
| Camera capture | 30-50ms |
| Fish-eye undistort (if GoPro) | 2-5ms |
| MediaPipe inference | 15-30ms |
| Angle extraction + smoothing | <1ms |
| Simulator step + render | 5-10ms |
| **Total** | **~55-95ms** |

---

## 3. Camera Setup

### Option A: GoPro as USB Webcam (RECOMMENDED for demo wow-factor)

**Supported models:** Hero 8 Black, 9, 10, 11, 12, 13 (all support USB webcam mode)

**Setup steps:**
1. Connect GoPro to computer via USB-C cable
2. On GoPro: Preferences → Connections → USB Connection → **GoPro Connect** (not MTP)
3. GoPro screen shows "GoPro Webcam" with a red dot
4. It appears as a standard UVC webcam — OpenCV sees it as a video device

**OpenCV capture:**
```python
# GoPro appears as a webcam device
cap = cv2.VideoCapture(0)  # or 1, 2 — find the right index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
```

**Latency:** ~100-150ms via USB webcam mode (acceptable for demo)

**Fish-eye correction (IMPORTANT for GoPro):**
GoPro wide-angle lens introduces barrel distortion. MediaPipe can still detect poses with mild distortion, but correction improves accuracy significantly at frame edges.

```python
import cv2
import numpy as np

# Approximate GoPro Hero wide-angle distortion coefficients
# These are reasonable starting values — calibrate with checkerboard for precision
GOPRO_MATRIX = np.array([
    [910, 0, 640],
    [0, 910, 360],
    [0, 0, 1]
], dtype=np.float32)

GOPRO_DIST = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

# Pre-compute undistort maps (do once at startup)
h, w = 720, 1280
new_matrix, roi = cv2.getOptimalNewCameraMatrix(GOPRO_MATRIX, GOPRO_DIST, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(GOPRO_MATRIX, GOPRO_DIST, None, new_matrix, (w, h), cv2.CV_16SC2)

def undistort_frame(frame):
    """Real-time undistortion using pre-computed maps (~2ms per frame)."""
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

**Key insight:** Using `cv2.remap()` with pre-computed maps is ~10x faster than `cv2.undistort()`. At 720p, expect ~2ms per frame — negligible.

**Does MediaPipe need pre-correction?** MediaPipe works okay with mild barrel distortion (the face/torso landmarks in center are fine), but arm landmarks at frame edges will have positional errors of 5-15% without correction. For a hackathon demo it's optional but recommended.

### Option B: iPhone via Continuity Camera (macOS only)

**Requirements:** macOS Ventura+, iPhone running iOS 16+, same Apple ID
**Setup:** iPhone automatically appears as a webcam in OpenCV when nearby
```python
cap = cv2.VideoCapture(0)  # Continuity Camera often takes index 0
```
**Latency:** ~60-100ms (lower than GoPro USB!)
**Pros:** No fish-eye distortion, great image quality
**Cons:** macOS only, need a mount for the phone

### Option C: Android via DroidCam

**Setup:** Install DroidCam on phone + DroidCam client on PC
**Capture:** DroidCam creates a virtual webcam device
```python
cap = cv2.VideoCapture(0)  # DroidCam virtual device
# OR via IP stream (higher latency):
cap = cv2.VideoCapture("http://PHONE_IP:4747/video")
```
**Latency:** ~80-120ms via USB tethering, ~150-300ms via WiFi
**Pros:** Works on Windows/Linux, free tier available
**Cons:** WiFi mode adds latency

### Option D: Laptop Webcam (FALLBACK — always works)

**For development/testing only.** Mount laptop facing you, extend arm forward.
```python
cap = cv2.VideoCapture(0)
```
**Latency:** ~30-50ms (lowest)
**Note:** Not egocentric, but perfect for developing the pipeline before mounting a camera

### Camera Selection Strategy
1. **Start development** with laptop webcam (Option D) — instant, no setup
2. **Switch to GoPro/phone** once pipeline works — for the true egocentric experience
3. **Demo day:** GoPro chest-mounted for wow factor, laptop webcam as backup

---

## 4. Pose Estimation Pipeline

### MediaPipe Pose Configuration

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,       # Video mode (uses tracking between frames)
    model_complexity=1,             # 0=lite, 1=full, 2=heavy. Use 1.
    smooth_landmarks=True,          # Built-in temporal smoothing
    enable_segmentation=False,      # Don't need body mask
    min_detection_confidence=0.5,   # Initial detection threshold
    min_tracking_confidence=0.5     # Frame-to-frame tracking threshold
)
```

### Key Landmarks for Arm Control

```
Right arm:  shoulder=12, elbow=14, wrist=16, index=20, thumb=22, pinky=18
Left arm:   shoulder=11, elbow=13, wrist=15, index=19, thumb=21, pinky=17
Torso ref:  right_hip=24, left_hip=23 (for shoulder angle reference)
```

Each landmark gives: `x` (0-1), `y` (0-1), `z` (relative depth), `visibility` (0-1)

### Visibility Gating (Critical for Egocentric)

```python
MIN_VISIBILITY = 0.5

def landmarks_visible(landmarks, indices):
    """Check if all required landmarks are sufficiently visible."""
    return all(landmarks[i].visibility > MIN_VISIBILITY for i in indices)

# Only update servos when arm is actually visible
RIGHT_ARM_INDICES = [12, 14, 16]  # shoulder, elbow, wrist
if landmarks_visible(lm, RIGHT_ARM_INDICES):
    # Extract angles and send to sim
    pass
else:
    # Hold last known position
    pass
```

### Hand Landmark Detection (for Gripper)

```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_gripper_openness(hand_landmarks):
    """Return 0.0 (closed/pinch) to 1.0 (fully open)."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return np.clip((dist - 0.02) / 0.13, 0.0, 1.0)
```

---

## 5. Joint Mapping Engine

### SO-ARM101 Joint Mapping

| SO-ARM101 Joint | Servo ID | Human Motion | MediaPipe Extraction |
|-----------------|----------|-------------|---------------------|
| shoulder_pan | 1 | Arm left/right swing | `atan2(elbow.x - shoulder.x, elbow.z - shoulder.z)` |
| shoulder_lift | 2 | Arm raise up/down | `angle(hip, shoulder, elbow)` |
| elbow_flex | 3 | Elbow bend | `angle(shoulder, elbow, wrist)` |
| wrist_flex | 4 | Wrist up/down | `angle(elbow, wrist, index_finger)` |
| wrist_roll | 5 | Forearm rotation | Estimate from wrist landmark positions (hard) |
| gripper | 6 | Hand open/close | Thumb-index finger tip distance |

### Geometric Angle Extraction

```python
import numpy as np

def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z])

def angle_3pts(a, b, c):
    """Angle at point b formed by vectors b→a and b→c, in radians."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def extract_joint_angles(landmarks):
    """Extract 6 joint angles from MediaPipe pose landmarks."""
    # Get key points
    r_shoulder = landmark_to_np(landmarks[12])
    r_elbow = landmark_to_np(landmarks[14])
    r_wrist = landmark_to_np(landmarks[16])
    r_hip = landmark_to_np(landmarks[24])
    r_index = landmark_to_np(landmarks[20])

    # 1. Shoulder pan: horizontal rotation of upper arm
    upper_arm = r_elbow - r_shoulder
    shoulder_pan = np.arctan2(upper_arm[0], -upper_arm[2])  # x/z plane

    # 2. Shoulder lift: angle between torso and upper arm
    shoulder_lift = angle_3pts(r_hip, r_shoulder, r_elbow)

    # 3. Elbow flex: angle at elbow
    elbow_flex = angle_3pts(r_shoulder, r_elbow, r_wrist)

    # 4. Wrist flex: angle at wrist (needs hand landmark)
    wrist_flex = angle_3pts(r_elbow, r_wrist, r_index)

    # 5. Wrist roll: approximation from forearm orientation
    # Use cross product of forearm with gravity to estimate roll
    forearm = r_wrist - r_elbow
    wrist_roll = np.arctan2(forearm[0], forearm[1])

    return {
        'shoulder_pan': shoulder_pan,
        'shoulder_lift': shoulder_lift,
        'elbow_flex': elbow_flex,
        'wrist_flex': wrist_flex,
        'wrist_roll': wrist_roll,
    }
```

### Angle-to-Action Mapping

```python
# gym-soarm action space is 6D: joint angles in radians
# Map human angles to robot angle ranges

JOINT_LIMITS = {
    'shoulder_pan':  (-np.pi * 0.75, np.pi * 0.75),  # ~270°
    'shoulder_lift': (0.0, np.pi),                      # ~180°
    'elbow_flex':    (0.0, np.pi),                      # ~180°
    'wrist_flex':    (-np.pi/2, np.pi/2),               # ~180°
    'wrist_roll':    (-np.pi, np.pi),                   # ~360°
    'gripper':       (0.0, 1.0),                        # normalized
}

def human_to_robot_action(joint_angles, gripper_openness):
    """Convert extracted human angles to gym-soarm action array."""
    action = np.zeros(6)

    # Map each joint with scaling and offset
    action[0] = np.clip(joint_angles['shoulder_pan'], *JOINT_LIMITS['shoulder_pan'])
    action[1] = np.clip(joint_angles['shoulder_lift'] - np.pi/2, *JOINT_LIMITS['shoulder_lift'])
    action[2] = np.clip(np.pi - joint_angles['elbow_flex'], *JOINT_LIMITS['elbow_flex'])
    action[3] = np.clip(joint_angles['wrist_flex'] - np.pi/2, *JOINT_LIMITS['wrist_flex'])
    action[4] = np.clip(joint_angles['wrist_roll'], *JOINT_LIMITS['wrist_roll'])
    action[5] = gripper_openness

    return action
```

### Smoothing Filter

```python
class ExponentialSmoother:
    """Exponential moving average filter for stable joint control."""

    def __init__(self, alpha=0.3, num_joints=6):
        self.alpha = alpha
        self.state = None

    def update(self, values):
        values = np.array(values)
        if self.state is None:
            self.state = values.copy()
        else:
            self.state = self.alpha * values + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self):
        self.state = None
```

**Tuning `alpha`:**
- `alpha=0.1` → very smooth, sluggish response (good for demo)
- `alpha=0.3` → balanced (recommended starting point)
- `alpha=0.6` → responsive but some jitter
- `alpha=1.0` → no smoothing (raw)

---

## 6. Simulator Integration

### Choice: gym-soarm (MuJoCo-based)

**Why gym-soarm over alternatives:**
| Criteria | gym-soarm | tuul-ai/so101_sim | lerobot-sim-teleop |
|----------|-----------|-------------------|-------------------|
| Setup time | ~2 min (`pip install`) | ~15 min (clone + install) | ~15 min |
| Set joint angles directly | ✅ `env.step(action)` | ✅ but more complex API | ✅ via keyboard/controller |
| Real-time viz | ✅ OpenCV viewer | ✅ MuJoCo viewer | ✅ Viser web UI |
| SO-ARM101 model | ✅ Exact | ✅ Exact | ✅ SO-100 |
| Gymnasium API | ✅ Standard | ✅ Compatible | ✅ Compatible |
| Multi-camera | ✅ 3 views | ❌ | ✅ |
| **Verdict** | **Best for real-time teleop** | Better for RL/IL | Better for data collection |

### gym-soarm Integration Code

```python
import gymnasium as gym
import gym_soarm

# Create environment
env = gym.make(
    'SoArm-v0',
    render_mode='human',           # Show visualization window
    obs_type='pixels_agent_pos',   # Get both camera images and joint positions
    camera_config='front_wrist'    # Two camera views
)

obs, info = env.reset()

# Main teleop loop
while True:
    # action is 6D: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    action = get_action_from_pose()  # Our pipeline
    obs, reward, terminated, truncated, info = env.step(action)

    # obs['agent_pos'] = current joint positions (6 values)
    # obs['pixels'] = camera images from sim
```

### Visualization Strategy

The demo will show a **split-screen** view:
```
┌─────────────────────────────┬─────────────────────────────┐
│                             │                             │
│   CAMERA FEED               │   SIMULATED ARM             │
│   (with MediaPipe overlay)  │   (gym-soarm MuJoCo view)   │
│                             │                             │
│   Skeleton drawn on frame   │   Robot mimics movements    │
│   Joint angles displayed    │   Multi-camera views        │
│                             │                             │
└─────────────────────────────┴─────────────────────────────┘
│              Joint Angle Dashboard (bottom bar)            │
│  Pan: -0.3  Lift: 1.2  Elbow: 0.8  Wrist: 0.1  Grip: 0.7│
└────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Timeline (Hour-by-Hour)

### ⏱️ MVP Phase (Hours 0-3) — Get Something Moving

**Hour 0-1: Camera + MediaPipe Pipeline**
- [ ] Set up project, install dependencies
- [ ] OpenCV capture from laptop webcam (fallback camera)
- [ ] MediaPipe Pose running, landmarks drawn on screen
- [ ] Verify arm landmarks detected when extending arm forward
- [ ] Print raw landmark coordinates to console
- **Milestone:** See skeleton overlay on live video ✅

**Hour 1-2: Joint Angle Extraction + Simulator**
- [ ] Install gym-soarm, verify `SoArm-v0` environment runs
- [ ] Implement `extract_joint_angles()` from landmarks
- [ ] Implement `ExponentialSmoother`
- [ ] Connect: landmarks → angles → `env.step(action)`
- [ ] First arm movement in simulator from body movement!
- **Milestone:** Move your arm, robot arm moves ✅

**Hour 2-3: Tuning + Stability**
- [ ] Tune angle mapping (scaling, offsets) for natural feel
- [ ] Add visibility gating (hold position when arm not visible)
- [ ] Tune smoothing alpha
- [ ] Add joint limit clamping
- [ ] Side-by-side visualization (camera feed + sim)
- **Milestone:** Stable, usable arm control ✅

### 🎨 Polish Phase (Hours 3-6) — Make It Demo-Ready

**Hour 3-4: Gripper + Wrist**
- [ ] Add MediaPipe Hands for gripper control
- [ ] Implement thumb-index pinch → gripper open/close
- [ ] Improve wrist flex/roll mapping
- [ ] Test pick-and-place motion in sim
- **Milestone:** 5-DOF + gripper control ✅

**Hour 4-5: GoPro/Phone Integration + Fish-eye**
- [ ] Switch from laptop webcam to GoPro USB or phone
- [ ] Implement fish-eye undistortion (if GoPro)
- [ ] Mount camera on chest, test egocentric tracking
- [ ] Tune parameters for egocentric viewpoint
- **Milestone:** True egocentric teleop working ✅

**Hour 5-6: Demo Polish**
- [ ] Build split-screen visualization with dashboard
- [ ] Add FPS counter, latency display
- [ ] Add joint angle readout overlay
- [ ] Record demo video
- [ ] Write demo script / talking points
- [ ] Clean up code, add comments
- **Milestone:** Demo-ready! ✅

---

## 8. File Structure

```
pov-arm-teleop/
├── README.md                    # Project overview
├── IMPLEMENTATION-PLAN.md       # This document
├── RESEARCH.md                  # Deep research report
├── requirements.txt             # Python dependencies
├── config.py                    # Joint limits, camera params, tuning constants
├── main.py                      # Main teleop loop (entry point)
├── camera.py                    # Camera capture + fish-eye correction
├── pose_tracker.py              # MediaPipe Pose + Hands wrapper
├── joint_mapper.py              # Landmark → joint angle extraction
├── smoother.py                  # Exponential smoothing filter
├── sim_controller.py            # gym-soarm environment wrapper
├── visualizer.py                # Split-screen display builder
├── calibrate_camera.py          # GoPro calibration tool (optional)
├── demo.py                      # Demo mode with pre-recorded fallback
├── research/                    # Research documents
│   ├── related-projects.md
│   ├── so-arm101-specs.md
│   ├── pose-estimation-comparison.md
│   └── egocentric-vision.md
└── assets/
    └── demo_recording.mp4       # Pre-recorded demo video
```

---

## 9. Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Core dependencies
pip install mediapipe>=0.10.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0

# Simulator
pip install gym-soarm
# gym-soarm pulls in: gymnasium, mujoco, dm-control

# Optional but recommended
pip install scipy           # For advanced filtering (Kalman)
```

**One-liner:**
```bash
pip install mediapipe opencv-python numpy gym-soarm scipy
```

**Environment variable (required for MuJoCo performance):**
```bash
export MUJOCO_GL='egl'      # Linux with GPU
# export MUJOCO_GL='glfw'   # Linux without GPU / macOS
```

---

## 10. Fallback Plans

| Problem | Fallback |
|---------|----------|
| **GoPro won't connect as webcam** | Use phone (DroidCam/Continuity Camera) or laptop webcam |
| **gym-soarm won't install** | Use MuJoCo directly with SO100 model from MuJoCo Menagerie: `pip install mujoco` and load MJCF from `mujoco_menagerie/google_robot` or write a simple PyBullet viz |
| **MediaPipe arm tracking too jittery** | Increase smoothing (alpha=0.15), use `model_complexity=2`, reduce camera resolution to 640x480 for faster inference |
| **Egocentric view doesn't work well** | Prop up camera on table facing you (third-person view) — pipeline works the same, just different mount |
| **Fish-eye correction too slow** | Skip it — MediaPipe works okay without correction, just less accurate at edges |
| **Can't get gripper working** | Map gripper to keyboard key (spacebar toggle) — still impressive demo without vision-based gripper |
| **Wrist roll unreliable** | Lock wrist roll to 0 (neutral) — 4-DOF + gripper is still a great demo |
| **Everything breaks** | Pre-record a video of it working, run sim with keyboard control from gym-soarm slider example |

---

## 11. Demo Script

### What Judges See (2-minute demo)

**Setup:** Laptop on table, GoPro mounted on chest harness, split-screen showing camera feed (left) and simulated arm (right).

**Script:**

1. **[0:00-0:15] Hook:** "What if you could control a robot arm just by moving your own arm? No VR headset, no special gloves — just a camera strapped to your chest."

2. **[0:15-0:30] Show the setup:** Point to chest-mounted camera, show the split screen. "The camera captures my arm movements from my own point of view. Computer vision extracts my joint angles in real-time."

3. **[0:30-1:00] Live demo — basic movements:**
   - Reach forward → robot reaches forward
   - Raise arm → robot raises
   - Bend elbow → robot bends
   - Sweep left/right → robot pans
   - "Every movement maps 1:1 to the robot's joints."

4. **[1:00-1:30] Live demo — manipulation:**
   - Open/close hand → gripper opens/closes
   - Mime picking up an object → robot does the same in sim
   - "The gripper responds to my hand — pinch to close, open to release."

5. **[1:30-1:50] Tech explanation:**
   - "MediaPipe extracts 33 body landmarks at 30 FPS. We compute joint angles geometrically and send them through a smoothing filter to the MuJoCo simulator. Total latency: under 100 milliseconds."

6. **[1:50-2:00] Vision:**
   - "This approach scales to any robot arm, needs zero special hardware, and could enable remote teleoperation over the internet. Imagine controlling a robot in a warehouse from your living room, using just your phone."

### Pre-Demo Checklist
- [ ] Camera connected and capturing
- [ ] MediaPipe detecting arms reliably
- [ ] Simulator window positioned next to camera feed
- [ ] Room lighting adequate (avoid backlight)
- [ ] Wear contrasting sleeves (not skin-tone colored)
- [ ] Test full range of motion before judges arrive
- [ ] Have laptop webcam fallback ready

---

## Appendix: Quick Reference

### MediaPipe Landmark IDs (Right Arm)
```
12 = right_shoulder
14 = right_elbow
16 = right_wrist
18 = right_pinky
20 = right_index
22 = right_thumb
24 = right_hip
```

### gym-soarm Action Space
```
action[0] = shoulder_pan    (radians)
action[1] = shoulder_lift   (radians)
action[2] = elbow_flex      (radians)
action[3] = wrist_flex      (radians)
action[4] = wrist_roll      (radians)
action[5] = gripper         (0.0-1.0)
```

### Useful Commands
```bash
# Run the main teleop
python main.py

# Run with specific camera index
python main.py --camera 1

# Run with GoPro fish-eye correction
python main.py --camera 0 --undistort

# Run sim-only mode (keyboard control for testing)
python examples/slider_control_final.py

# Calibrate GoPro (print checkerboard, capture ~20 images)
python calibrate_camera.py
```
