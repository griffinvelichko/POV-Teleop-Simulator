"""
smoother.py — Signal smoothing filters for stable joint control.

The Smoother applies an EMA to each joint independently.
Tuning alpha:
  - 0.1 = very smooth, sluggish (good for demos)
  - 0.3 = balanced (default)
  - 0.6 = responsive, some jitter
  - 1.0 = no smoothing (raw)
"""

import numpy as np

from config import SMOOTHING_ALPHA, DEADBAND_THRESHOLD


class Smoother:
    """
    Exponential Moving Average filter with optional deadband.

    Usage:
        s = Smoother(alpha=0.3)
        smoothed = s.update(raw_values)  # np.ndarray → np.ndarray
        s.reset()
    """

    def __init__(self, alpha=SMOOTHING_ALPHA, num_joints=6):
        self.alpha = alpha
        self.num_joints = num_joints
        self.state = None
        self.deadband = DEADBAND_THRESHOLD

    def update(self, values: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to a joint value array.

        Args:
            values: np.ndarray of shape (num_joints,)

        Returns:
            np.ndarray of shape (num_joints,) — smoothed values
        """
        values = np.asarray(values, dtype=np.float64).flatten()[: self.num_joints]
        if values.size != self.num_joints:
            values = np.resize(values, self.num_joints)

        if self.state is None:
            self.state = values.copy()
            return self.state.copy()

        raw_smoothed = self.alpha * values + (1.0 - self.alpha) * self.state
        delta = np.abs(raw_smoothed - self.state)
        mask = delta > self.deadband
        self.state = np.where(mask, raw_smoothed, self.state)

        return self.state.copy()

    def reset(self) -> None:
        """Reset the filter state. Next update() will initialize from scratch."""
        self.state = None


if __name__ == "__main__":
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
