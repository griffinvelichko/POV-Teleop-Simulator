#!/usr/bin/env bash
# POV-Teleop-Simulator setup: create venv, install MuJoCo, install project deps from pyproject.toml.
# MuJoCo 2.x needs Python 3.9–3.12 (prebuilt wheels); Python 3.14+ would build from source and fail.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="${VENV_DIR:-.venv}"

# Prefer Python 3.9–3.12 so MuJoCo gets a prebuilt wheel (no MUJOCO_PATH / source build).
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$candidate" &>/dev/null; then
    if "$candidate" -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info < (3, 14) else 1)" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: No Python 3.9–3.12 found. MuJoCo 2.x needs a supported version (not 3.14+)."
  echo "Install one of: python3.12, python3.11, python3.10, or python3.9 (e.g. brew install python@3.12)"
  exit 1
fi

echo "==> Using Python: $PYTHON ($($PYTHON --version))"
echo "==> Creating virtual environment at $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install -U pip

# On macOS, mujoco 2.3.7 has no prebuilt wheel for arm64; pip builds from source and requires
# MUJOCO_PATH (and MUJOCO_PLUGIN_PATH). Download the official binary and point to it.
MUJOCO_VERSION="2.3.7"
if [[ "$(uname -s)" == "Darwin" ]]; then
  MUJOCO_INSTALL_DIR="$SCRIPT_DIR/.mujoco_install"
  NEED_MUJOCO_BINARY=false
  MUJOCO_APP=$(find "$MUJOCO_INSTALL_DIR" -maxdepth 1 -name "*.app" -type d 2>/dev/null | head -1)
  if [[ -n "$MUJOCO_APP" ]]; then
    MUJOCO_FRAMEWORKS="${MUJOCO_APP}/Contents/Frameworks"
    [[ ! -d "$MUJOCO_FRAMEWORKS/mujoco.framework" ]] && NEED_MUJOCO_BINARY=true
  else
    NEED_MUJOCO_BINARY=true
  fi

  if [[ "$NEED_MUJOCO_BINARY" == true ]]; then
    echo "==> Downloading MuJoCo $MUJOCO_VERSION binary for macOS (required for source build)"
    MUJOCO_DMG="$SCRIPT_DIR/mujoco-${MUJOCO_VERSION}-macos-universal2.dmg"
    MUJOCO_URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-macos-universal2.dmg"
    if ! curl -sSfL -o "$MUJOCO_DMG" "$MUJOCO_URL"; then
      echo "ERROR: Failed to download MuJoCo from $MUJOCO_URL"
      exit 1
    fi
    echo "==> Installing MuJoCo binary to $MUJOCO_INSTALL_DIR"
    mkdir -p "$MUJOCO_INSTALL_DIR"
    MOUNT_POINT="$SCRIPT_DIR/.mujoco_mount"
    mkdir -p "$MOUNT_POINT"
    hdiutil attach "$MUJOCO_DMG" -nobrowse -mountpoint "$MOUNT_POINT" -quiet
    if [[ -d "$MOUNT_POINT/MuJoCo.app" ]]; then
      cp -R "$MOUNT_POINT/MuJoCo.app" "$MUJOCO_INSTALL_DIR/"
    else
      APP_PATH=$(find "$MOUNT_POINT" -maxdepth 2 -name "*.app" -type d 2>/dev/null | head -1)
      if [[ -n "$APP_PATH" ]]; then
        cp -R "$APP_PATH" "$MUJOCO_INSTALL_DIR/"
      else
        echo "ERROR: No MuJoCo.app found in DMG. Contents:"
        ls -la "$MOUNT_POINT"
        hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null || true
        rm -f "$MUJOCO_DMG"
        exit 1
      fi
    fi
    hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null || true
    rm -f "$MUJOCO_DMG"
    rmdir "$MOUNT_POINT" 2>/dev/null || true
    MUJOCO_APP=$(find "$MUJOCO_INSTALL_DIR" -maxdepth 1 -name "*.app" -type d 2>/dev/null | head -1)
    MUJOCO_FRAMEWORKS="${MUJOCO_APP}/Contents/Frameworks"
  fi

  if [[ -d "$MUJOCO_FRAMEWORKS/mujoco.framework" ]]; then
    export MUJOCO_PATH="$MUJOCO_FRAMEWORKS"
    export MUJOCO_PLUGIN_PATH="$MUJOCO_FRAMEWORKS"
    echo "==> Using MuJoCo at MUJOCO_PATH=$MUJOCO_PATH"
    if ! command -v cmake &>/dev/null; then
      echo "ERROR: CMake is required to build MuJoCo from source on macOS. Install with:"
      echo "  brew install cmake"
      exit 1
    fi
  fi
fi

echo "==> Installing MuJoCo"
pip install "mujoco>=2.3.7,<3.0.0"

echo "==> Installing project and dependencies from pyproject.toml"
pip install -e .

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "For rendering, set one of:"
echo "  export MUJOCO_GL=egl   # Linux with GPU"
echo "  export MUJOCO_GL=glfw  # macOS or Linux without GPU"
echo ""
