# Hackathon Sprint Structure

Time-boxed build. 4 developers. 4 blocks. 3 checkpoints.

## Schedule

| Block | Time | Focus | Exit Criteria |
|-------|------|-------|---------------|
| **0: Setup** | 0:00–0:30 | Git sync, venv, install deps, download models, confirm module ownership | Everyone can import their dependencies and run standalone tests |
| **1: Infrastructure** | 0:30–2:00 | Camera capture, MediaPipe wrapper, joint mapping skeleton, sim environment | Each module runs standalone with `if __name__ == "__main__"` |
| **Checkpoint 1** | 2:00 | Quick sync: do interfaces match? Codex review of contracts | |
| **2: Core Logic** | 2:00–3:30 | Angle extraction math, smoothing, pose-to-action pipeline, display layout | Camera → Pose → Mapper produces valid 6D actions |
| **Checkpoint 2** | 3:30 | Integration test: camera → pose → mapper → simulator moves | |
| **3: Demo Pipeline** | 3:30–5:00 | Full pipeline integration, split-screen display, gripper control, polish | main.py runs end-to-end with split-screen view |
| **Checkpoint 3** | 5:00 | End-to-end test of full demo flow | |
| **4: Polish** | 5:00–6:00 | Demo prep, presentation, edge case fixes, FPS optimization | Ready to present |

## Module Ownership (Default)

| Developer | Primary Files | Block 1 Focus |
|-----------|--------------|---------------|
| **Griffin** | `config.py`, `camera.py`, `requirements.txt` | Camera capture, fish-eye correction, shared constants |
| **Torrin** | `pose.py` | MediaPipe Tasks API wrapper (PoseLandmarker + HandLandmarker) |
| **Damian** | `mapping.py`, `smoother.py` | Joint angle extraction math, EMA smoothing filter |
| **Jaden** | `sim.py`, `display.py`, `main.py` | gym-soarm environment wrapper, visualization skeleton |

These are starting points. Shift as needed — communicate when you do.

**Integration note**: Data flows left to right: Griffin's camera → Torrin's pose tracker → Damian's joint mapper → Jaden's simulator. Each person's output is the next person's input. The interfaces in `docs/specs/interface_contracts.md` define exactly how they connect.

## Checkpoint Protocol

Each checkpoint is a ~5 minute sync:

1. **Stand up**: What's done, what's blocked, what's next
2. **Integration check**: Does my output match your input?
3. **Codex review** (optional): If interfaces feel shaky, run a quick Codex check
4. **Adjust plan**: Re-allocate work if someone is ahead/behind

If you're behind schedule, skip the Codex review. The sync is non-negotiable.

## Coordination Rules

- **Never block on a teammate.** If you need their module, stub the interface and keep going.
- **Merge to main frequently.** Every 30-60 minutes minimum.
- **Shout conflicts immediately.** Don't silently resolve merge conflicts.
- **Specs are pre-loaded.** ADRs, interface contracts, and individual plans are in `docs/`. Read them before building.
- **If something breaks main, fix it or revert within 5 minutes.**

## Stubs

When you need a teammate's module that doesn't exist yet:

```python
# stub — replace when Torrin's pose tracker lands
class PoseTracker:
    def process(self, bgr_frame, timestamp_ms):
        return PoseResult(pose_world_landmarks=None, pose_landmarks=None,
                         hand_landmarks=None, hand_world_landmarks=None,
                         timestamp_ms=timestamp_ms)
```

Mark stubs clearly. They'll get replaced at integration.

## Demo Prep (Block 4)

The last hour is sacred. No new features. Only:
- Bug fixes that affect the demo flow
- FPS optimization if running below 20 FPS
- Presentation talking points
- One full dry run
