import json
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pybullet as p


def get_link_world_position(body_id: int, link_idx: int) -> Tuple[float, float, float]:
    ls = p.getLinkState(body_id, link_idx)
    return tuple(ls[0])


def get_body_position(body_id: int) -> Tuple[float, float, float]:
    pos, orn = p.getBasePositionAndOrientation(body_id)
    return tuple(pos)


def l2_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def path_radial_error_and_coverage(path_xyz: List[Tuple[float, float, float]], center: Tuple[float, float, float], radius: float) -> Tuple[float, float]:
    if not path_xyz:
        return float("inf"), 0.0
    center_xy = np.array(center[:2])
    angles: List[float] = []
    radial_errors: List[float] = []
    for pt in path_xyz:
        v = np.array(pt[:2]) - center_xy
        r = np.linalg.norm(v)
        radial_errors.append(abs(r - radius))
        angles.append(math.atan2(v[1], v[0]))
    # Normalize angle coverage
    angles_sorted = np.unwrap(np.array(sorted(angles)))
    coverage = (angles_sorted[-1] - angles_sorted[0]) * 180.0 / math.pi
    mean_radial_error = float(np.mean(radial_errors))
    return mean_radial_error, float(coverage)


def write_results_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)