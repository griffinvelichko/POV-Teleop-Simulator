"""
config.py — Shared configuration constants for the POV teleop pipeline.

All tunable parameters live here. No other file should hardcode these values.
Import from here: from config import CAMERA_WIDTH, JOINT_LIMITS, ...
"""

import numpy as np

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_DEVICE = 0  # OpenCV device index (0 = default webcam)
CAMERA_WIDTH = 1280  # Capture resolution width
CAMERA_HEIGHT = 720  # Capture resolution height
CAMERA_FPS = 30  # Target framerate

# ──────────────────────────────────────────────
# MediaPipe Models
# ──────────────────────────────────────────────
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

# ──────────────────────────────────────────────
# Pose Estimation Thresholds
# ──────────────────────────────────────────────
MIN_POSE_CONFIDENCE = 0.5  # Minimum detection confidence for pose
MIN_TRACKING_CONFIDENCE = 0.5  # Frame-to-frame tracking threshold
MIN_VISIBILITY = 0.5  # Minimum per-landmark visibility to trust

# Landmark indices (MediaPipe Pose, 33-landmark model)
# Right arm chain
LM_RIGHT_SHOULDER = 12
LM_RIGHT_ELBOW = 14
LM_RIGHT_WRIST = 16
LM_RIGHT_INDEX = 20
LM_RIGHT_THUMB = 22
LM_RIGHT_PINKY = 18
LM_RIGHT_HIP = 24

# Left arm chain (for future use)
LM_LEFT_SHOULDER = 11
LM_LEFT_ELBOW = 13
LM_LEFT_WRIST = 15
LM_LEFT_INDEX = 19
LM_LEFT_HIP = 23

# Which arm to track
TRACK_RIGHT_ARM = True

# Camera perspective: MediaPipe pose models assume third-person (camera facing the user).
# When True, mapping inverts horizontal angles so robot motion matches the camera view.
THIRD_PERSON_VIEW = True

# Required landmarks that must be visible for arm tracking
REQUIRED_LANDMARKS = [
    LM_RIGHT_SHOULDER,
    LM_RIGHT_ELBOW,
    LM_RIGHT_WRIST,
    LM_RIGHT_HIP,
]

# ──────────────────────────────────────────────
# Joint Limits (from actual SO-ARM101 MJCF — do NOT change these)
# ──────────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_LIMITS = {
    "shoulder_pan": (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex": (-1.690, 1.690),
    "wrist_flex": (-1.658, 1.658),
    "wrist_roll": (-2.744, 2.841),
    "gripper": (-0.175, 1.745),
}

# Home position (from MJCF keyframe)
HOME_POSITION = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

# ──────────────────────────────────────────────
# Smoothing
# ──────────────────────────────────────────────
SMOOTHING_ALPHA = 0.3  # EMA alpha: 0.1=smooth/sluggish, 0.3=balanced, 0.6=responsive
DEADBAND_THRESHOLD = 0.02  # Ignore joint changes smaller than this (radians)

# ──────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────
SIM_RENDER_MODE = "human"
SIM_OBS_TYPE = "pixels_agent_pos"
# POV camera: behind the robot looking forward over its shoulder
SIM_CAMERA_CONFIG = "pov"

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
DISPLAY_WIDTH = 1280  # Total display window width
DISPLAY_HEIGHT = 520  # Total display window height (480 video + 40 dashboard)
WINDOW_NAME = "POV Teleop"
