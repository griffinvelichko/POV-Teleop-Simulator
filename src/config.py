"""
config.py — Shared configuration constants for the POV teleop pipeline.

All tunable parameters live here. No other file should hardcode these values.
Owner: Griffin (see PLAN-GRIFFIN.md). Created for pipeline integration.
"""

import numpy as np

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_DEVICE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
ENABLE_UNDISTORT = False

GOPRO_CAMERA_MATRIX = np.array([
    [910.0, 0.0, 640.0],
    [0.0, 910.0, 360.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

GOPRO_DIST_COEFFS = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

# ──────────────────────────────────────────────
# MediaPipe Models
# ──────────────────────────────────────────────
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

# ──────────────────────────────────────────────
# Pose Estimation Thresholds
# ──────────────────────────────────────────────
MIN_POSE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MIN_VISIBILITY = 0.5

# Landmark indices (MediaPipe Pose, 33-landmark model)
LM_RIGHT_SHOULDER = 12
LM_RIGHT_ELBOW = 14
LM_RIGHT_WRIST = 16
LM_RIGHT_INDEX = 20
LM_RIGHT_THUMB = 22
LM_RIGHT_PINKY = 18
LM_RIGHT_HIP = 24

LM_LEFT_SHOULDER = 11
LM_LEFT_ELBOW = 13
LM_LEFT_WRIST = 15
LM_LEFT_INDEX = 19
LM_LEFT_HIP = 23

TRACK_RIGHT_ARM = True

# Required for arm + wrist angle extraction (index needed for wrist_flex)
REQUIRED_LANDMARKS = [
    LM_RIGHT_SHOULDER,
    LM_RIGHT_ELBOW,
    LM_RIGHT_WRIST,
    LM_RIGHT_HIP,
    LM_RIGHT_INDEX,
]

# ──────────────────────────────────────────────
# Joint Limits (SO-ARM101 MJCF)
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
    "shoulder_pan":  (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex":    (-1.690, 1.690),
    "wrist_flex":    (-1.658, 1.658),
    "wrist_roll":    (-2.744, 2.841),
    "gripper":       (-0.175, 1.745),
}

HOME_POSITION = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

# ──────────────────────────────────────────────
# Smoothing (Damian: smoother.py)
# ──────────────────────────────────────────────
SMOOTHING_ALPHA = 0.3
DEADBAND_THRESHOLD = 0.02

# ──────────────────────────────────────────────
# Simulator / Display
# ──────────────────────────────────────────────
SIM_RENDER_MODE = "human"
SIM_OBS_TYPE = "pixels_agent_pos"
SIM_CAMERA_CONFIG = "front_wrist"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 520
WINDOW_NAME = "POV Teleop"
