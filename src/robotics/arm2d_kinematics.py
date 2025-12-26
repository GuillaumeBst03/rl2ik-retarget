from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Arm2DParams:
    """Parameters for a planar serial arm."""
    link_lengths: np.ndarray  # shape (dof,)

    def __post_init__(self):
        ll = np.asarray(self.link_lengths, dtype=np.float64)
        if ll.ndim != 1 or ll.size == 0:
            raise ValueError("link_lengths must be a 1D non-empty array.")
        if np.any(ll <= 0):
            raise ValueError("All link lengths must be > 0.")
        object.__setattr__(self, "link_lengths", ll)


def forward_kinematics(params: Arm2DParams, q: np.ndarray) -> np.ndarray:
    """
    Compute end-effector (x, y) position for a planar serial arm.

    Args:
        params: Arm2DParams
        q: joint angles in radians, shape (dof,)

    Returns:
        ee_xy: np.ndarray shape (2,)
    """
    q = np.asarray(q, dtype=np.float64)
    dof = params.link_lengths.size
    if q.shape != (dof,):
        raise ValueError(f"q must have shape ({dof},), got {q.shape}")

    # Cumulative angle for each link
    theta = np.cumsum(q)

    # Sum link vectors
    x = np.sum(params.link_lengths * np.cos(theta))
    y = np.sum(params.link_lengths * np.sin(theta))
    return np.array([x, y], dtype=np.float64)


def joint_positions(params: Arm2DParams, q: np.ndarray) -> np.ndarray:
    """
    Compute positions of all joints (including base and end-effector).

    Returns:
        pts: shape (dof+1, 2)
             pts[0] = base (0,0)
             pts[i] = end of link i (1-indexed) for i=1..dof
    """
    q = np.asarray(q, dtype=np.float64)
    dof = params.link_lengths.size
    if q.shape != (dof,):
        raise ValueError(f"q must have shape ({dof},), got {q.shape}")

    pts = np.zeros((dof + 1, 2), dtype=np.float64)
    theta = 0.0
    x, y = 0.0, 0.0
    for i in range(dof):
        theta += q[i]
        x += params.link_lengths[i] * np.cos(theta)
        y += params.link_lengths[i] * np.sin(theta)
        pts[i + 1] = (x, y)
    return pts


def sample_random_configuration(
    dof: int,
    low: float = -np.pi,
    high: float = np.pi,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(dof,)).astype(np.float64)