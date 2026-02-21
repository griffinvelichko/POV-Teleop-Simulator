# Related Projects & Papers

## Open Source Teleoperation Projects

### LeRobot (Hugging Face)
- **What:** Full framework for robot learning — data collection, training, deployment
- **Teleoperation:** Uses physical leader arm (1:1 kinematic match)
- **Relevance:** This is what SO-ARM101 natively uses. Could extend it with vision-based input
- **Link:** https://github.com/huggingface/lerobot
- **Key insight:** LeRobot's `send_action()` API accepts joint position dicts — we can feed it positions from our vision pipeline instead of from a leader arm

### GELLO
- **What:** Low-cost teleoperation using a 3D-printed replica arm as controller
- **Approach:** Physical leader arm with Dynamixel servos, reads joint positions
- **Relevance:** Shows the leader-follower paradigm; we're replacing the physical leader with vision
- **Link:** https://github.com/wuphilipp/gello_software
- **Paper:** https://arxiv.org/abs/2309.13037

### AnyTeleop / dex-retargeting
- **What:** Vision-based teleoperation using hand pose estimation
- **Approach:** Detects human hand pose → retargets to robot hand/arm
- **Relevance:** Closest to what we want, but focused on dexterous hands
- **Link:** https://github.com/dexsuite/dex-retargeting
- **Paper:** https://arxiv.org/abs/2307.04577

### Bunny-VisionPro
- **What:** Apple Vision Pro → bimanual robot teleoperation
- **Approach:** Uses Vision Pro's hand/wrist tracking → maps to robot arms
- **Relevance:** Similar concept but uses VR headset instead of GoPro
- **Link:** https://github.com/Dingry/BunnyVisionPro
- **Paper:** https://arxiv.org/abs/2407.03162

### Open-TeleVision
- **What:** Immersive teleoperation with stereo vision
- **Approach:** VR headset + hand tracking → humanoid robot control
- **Relevance:** High-end version of what we're building
- **Paper:** https://arxiv.org/abs/2407.01512

### UMI (Universal Manipulation Interface)
- **What:** Data collection interface using handheld grippers with cameras
- **Approach:** Not teleoperation per se, but captures manipulation data
- **Relevance:** Shows how to map human actions to robot actions
- **Link:** https://github.com/real-stanford/universal_manipulation_interface
- **Paper:** https://arxiv.org/abs/2402.10329

### ACE (Action Chunking with Egocentric observations)
- **What:** Policy learning from egocentric camera observations
- **Approach:** Trains policies that take egocentric images as input
- **Relevance:** Uses egocentric view but for learned policies, not direct teleoperation
- **Paper:** https://arxiv.org/abs/2408.11451

## MediaPipe + Robot Arm Projects on GitHub
Many hobby projects exist connecting MediaPipe to servo arms:
- Search: "mediapipe robot arm python" on GitHub
- Common pattern: MediaPipe → angle extraction → Arduino/serial → servos
- Most use third-person camera (webcam facing the person)
- Our innovation: doing it from first-person (egocentric) view

## Key Differentiator of Our Project
Most existing work uses either:
1. **Physical leader arm** (GELLO, LeRobot default)
2. **VR headset** (Bunny-VisionPro, Open-TeleVision)
3. **Third-person camera** (most MediaPipe + robot projects)

Our approach: **egocentric (first-person) camera + direct vision-based control**
- No extra hardware beyond a GoPro
- No VR headset needed
- Operator sees naturally, robot mimics
