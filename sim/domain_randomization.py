import numpy as np
import pybullet as p
from typing import Tuple


def randomize_masses(body_ids, scale_range: Tuple[float, float] = (0.8, 1.2), client_id: int = 0):
    rng = np.random.default_rng()
    for b in body_ids:
        dyn = p.getDynamicsInfo(b, -1, physicsClientId=client_id)
        mass = dyn[0]
        new_mass = float(mass * rng.uniform(scale_range[0], scale_range[1]))
        p.changeDynamics(b, -1, mass=new_mass, physicsClientId=client_id)


def randomize_friction(body_ids, lateral_range: Tuple[float, float] = (0.6, 1.2), client_id: int = 0):
    rng = np.random.default_rng()
    for b in body_ids:
        mu = rng.uniform(*lateral_range)
        p.changeDynamics(b, -1, lateralFriction=float(mu), physicsClientId=client_id)


def randomize_camera(jitter_xy: float = 0.05, jitter_z: float = 0.05) -> Tuple[float, float, float]:
    rng = np.random.default_rng()
    dx = rng.uniform(-jitter_xy, jitter_xy)
    dy = rng.uniform(-jitter_xy, jitter_xy)
    dz = rng.uniform(-jitter_z, jitter_z)
    return dx, dy, dz