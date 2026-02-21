"""
smoother.py — Signal smoothing filters for stable joint control.

The Smoother applies an Exponential Moving Average (EMA) to each joint
independently, with an optional deadband to suppress micro-jitter.

Tuning alpha:
  - 0.1 = very smooth, sluggish (good for demos)
  - 0.3 = balanced (default)
  - 0.6 = responsive, some jitter
  - 1.0 = no smoothing (raw)
"""

import numpy as np

from config import DEADBAND_THRESHOLD, SMOOTHING_ALPHA


class Smoother:
    """
    Exponential Moving Average filter with optional deadband.

    Usage:
        s = Smoother(alpha=0.3)
        smoothed = s.update(raw_values)  # np.ndarray -> np.ndarray
        s.reset()
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA, num_joints: int = 6):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if num_joints < 1:
            raise ValueError("num_joints must be >= 1")
        self.alpha = float(alpha)
        self.num_joints = int(num_joints)
        self.state: np.ndarray | None = None
        self.deadband = float(DEADBAND_THRESHOLD)

    def update(self, values: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to a joint value array.

        Args:
            values: np.ndarray of shape (num_joints,) or (6,). Dtype is coerced to float64.

        Returns:
            np.ndarray of shape (num_joints,) — smoothed values. New array each call.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("values must be 1D")
        if len(values) != self.num_joints:
            raise ValueError(
                f"values length {len(values)} != num_joints {self.num_joints}"
            )

        if self.state is None:
            self.state = values.copy()
            return self.state.copy()

        raw_smoothed = (
            self.alpha * values + (1.0 - self.alpha) * self.state
        )
        delta = np.abs(raw_smoothed - self.state)
        mask = delta > self.deadband
        self.state = np.where(mask, raw_smoothed, self.state)

        return self.state.copy()

    def reset(self) -> None:
        """Reset the filter state. Next update() will initialize from scratch."""
        self.state = None


if __name__ == "__main__":
    # Test: feed noisy data through smoother, verify output is smooth
    s = Smoother(alpha=0.3)

    np.random.seed(42)
    target = np.array([0.5, -1.0, 1.2, 0.3, 0.0, 0.5])

    print("Testing smoother with noisy data:")
    for i in range(20):
        noise = np.random.normal(0, 0.1, size=6)
        raw = target + noise
        smoothed = s.update(raw)
        print(f"  Step {i:2d}: raw[0]={raw[0]:+.3f}  smooth[0]={smoothed[0]:+.3f}")

    print(f"\nFinal state: {s.state}")
    print(f"Target was:  {target}")
    print(f"Error:       {np.abs(s.state - target).mean():.4f} (should be small)")
