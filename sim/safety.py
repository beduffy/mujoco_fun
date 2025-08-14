import numpy as np
from typing import Tuple


def clamp_joint_positions(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(q, lower, upper)


def clamp_joint_velocities(qdot: np.ndarray, vmax: float) -> np.ndarray:
    return np.clip(qdot, -abs(vmax), abs(vmax))


def rate_limit(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - prev
    norm = np.linalg.norm(delta)
    if norm <= max_step or norm < 1e-9:
        return target
    return prev + delta * (max_step / norm)