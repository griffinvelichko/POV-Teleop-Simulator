#!/usr/bin/env bash
# POV-Teleop-Simulator setup: create venv, install MuJoCo, install project deps from pyproject.toml.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Using Python: $(which python3 2>/dev/null || which python)"
python3 --version 2>/dev/null || python --version

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR" 2>/dev/null || python -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install -U pip

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
