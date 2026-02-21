# Pose Estimation Framework Comparison

## For Hackathon: MediaPipe Wins

| Framework | Speed (FPS) | GPU Required | Setup Complexity | Arm Tracking Quality | Hackathon Score |
|-----------|-------------|--------------|-----------------|---------------------|-----------------|
| **MediaPipe Pose** | 30-60 | No | ⭐ Easy | ⭐⭐⭐ Good | 🥇 Best |
| **MediaPipe Holistic** | 20-40 | No | ⭐ Easy | ⭐⭐⭐ Good + hands | 🥈 Great |
| ViTPose | 15-30 | Yes | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Excellent | 🥉 Overkill |
| HRNet | 10-25 | Yes | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Excellent | Not worth it |
| OpenPose | 10-20 | Yes | ⭐⭐ Medium | ⭐⭐⭐ Good | Too slow |
| MMPose | 15-30 | Yes | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Excellent | Too complex |

## MediaPipe Pose Landmarks (33 total)

### Relevant for Arm Control:
- **11** — Left Shoulder
- **12** — Right Shoulder  
- **13** — Left Elbow
- **14** — Right Elbow
- **15** — Left Wrist
- **16** — Right Wrist
- **17** — Left Pinky
- **18** — Right Pinky
- **19** — Left Index
- **20** — Right Index
- **21** — Left Thumb
- **22** — Right Thumb
- **23** — Left Hip (for torso reference)
- **24** — Right Hip (for torso reference)

### Each landmark provides:
- `x` — normalized 0-1 (horizontal)
- `y` — normalized 0-1 (vertical)
- `z` — relative depth (smaller = closer to camera)
- `visibility` — 0-1 confidence score

## Model Complexity Options
- `model_complexity=0` — Lite (fastest, least accurate)
- `model_complexity=1` — Full (good balance) ← **RECOMMENDED**
- `model_complexity=2` — Heavy (most accurate, slower)

## MediaPipe Hand Landmarks (21 per hand)
If using Holistic or separate Hand solution:
- 4 landmarks per finger (MCP, PIP, DIP, TIP)
- 1 wrist landmark
- Useful for: gripper control (measure hand openness)

### Gripper Control via Hand Landmarks
```python
# Distance between thumb tip (4) and index tip (8)
thumb = hand_landmarks[4]
index = hand_landmarks[8]
distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)

# Map to gripper: close when pinching, open when spread
gripper_pos = int(np.interp(distance, [0.02, 0.15], [3000, 1024]))
```
