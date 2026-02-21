# POV → SO-ARM101 Teleoperation

Control a **simulated** SO-ARM101 robotic arm using a body-mounted camera and computer vision — no physical robot needed.

**Concept:** Wear a GoPro/phone on your chest → MediaPipe tracks your arm → simulated robot arm mirrors your movements in real-time.

## How It Works
```
Camera (chest-mounted) → OpenCV → MediaPipe Pose → Joint Angles → gym-soarm Simulator
```

1. **Camera** captures first-person view of your arms
2. **MediaPipe Pose** extracts 33 body landmarks at 30 FPS
3. **Geometric extraction** computes shoulder, elbow, wrist angles
4. **Smoothing filter** stabilizes the signal
5. **gym-soarm** (MuJoCo) renders the SO-ARM101 moving in real-time

## Quick Start
```bash
pip install mediapipe opencv-python numpy gym-soarm scipy
export MUJOCO_GL='egl'  # or 'glfw' on macOS
python src/main.py      # or: python -m src.main (from project root)
```

## Documentation
- 📋 [Implementation Plan](IMPLEMENTATION-PLAN.md) — hour-by-hour build guide
- 🔬 [Research Report](RESEARCH.md) — deep technical research
- 🦾 [SO-ARM101 Specs](research/so-arm101-specs.md)
- 👁️ [Egocentric Vision](research/egocentric-vision.md)
- 🏋️ [Pose Estimation Comparison](research/pose-estimation-comparison.md)
- 🔗 [Related Projects](research/related-projects.md)

## Status
🚧 Implementation phase — hackathon weekend project (Feb 2026)

## License
MIT
