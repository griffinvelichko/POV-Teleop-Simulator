# Egocentric (First-Person) Vision for Arm Tracking

## The Core Challenge

Egocentric pose estimation is fundamentally different from third-person:
- **Partial body visibility** — you can only see your own arms/hands, not your full body
- **Self-occlusion** — arms cross, hands overlap
- **Unusual viewpoint** — looking down at your own hands is a rare perspective in training data
- **Motion blur** — head-mounted cameras move with the wearer
- **Depth ambiguity** — monocular camera can't determine absolute depth

## What MediaPipe Does Well (Egocentric)
- Tracks arms/hands when they're in the forward-reaching workspace
- Provides reasonable 3D estimates even from monocular camera
- Real-time performance (30+ FPS)
- Per-landmark visibility/confidence scores

## What MediaPipe Struggles With (Egocentric)
- Arms at sides (out of FOV)
- Arms behind the camera
- Close-up hand poses (too close to camera)
- When only fingertips are visible
- Extreme arm angles (reaching overhead from chest cam)

## Key Research Papers

### Ego4D (Meta, 2022)
- Massive egocentric video dataset (3,670 hours)
- Tasks include hand-object interaction, social interaction, episodic memory
- Relevant sub-tasks: hand tracking, object manipulation
- Paper: https://arxiv.org/abs/2110.07058
- Data: https://ego4d-data.org/

### Ego-Exo4D (Meta, 2024)
- Paired egocentric + exocentric (third-person) video
- Enables learning cross-view correspondence
- Paper: https://arxiv.org/abs/2311.18259

### EgoBody (ETH Zurich, 2022)
- Full body pose estimation from egocentric views
- Uses head-mounted camera + ground truth from motion capture
- Paper: https://arxiv.org/abs/2112.07445

### UnrealEgo (2022)
- Synthetic data for egocentric pose estimation
- Uses Unreal Engine to render first-person views with GT poses
- Paper: https://arxiv.org/abs/2208.01633

### xR-EgoPose (Facebook Reality Labs, 2020)
- Egocentric pose estimation for XR applications
- Paper: https://arxiv.org/abs/1907.10045

## Practical Recommendations for Hackathon

1. **Don't train custom models** — use MediaPipe out of the box
2. **Chest mount > head mount** — more stable arm visibility
3. **Wide-angle lens helps** — GoPro's wide FOV is actually an advantage
4. **Constrain the task** — only teleoperate when arms are in view
5. **Use visibility scores** — MediaPipe gives 0-1 visibility per landmark
6. **Implement deadzone** — don't update servo if landmark confidence < 0.5
7. **Kalman filter or EMA** — smooth out the jitter from inconsistent detections
