import numpy as np
from typing import List, Tuple


def segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    tmin, tmax = 0.0, 1.0
    d = p1 - p0
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if p0[i] < aabb_min[i] or p0[i] > aabb_max[i]:
                return False
        else:
            ood = 1.0 / d[i]
            t1 = (aabb_min[i] - p0[i]) * ood
            t2 = (aabb_max[i] - p0[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def rrt(start: np.ndarray, goal: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], max_iters: int = 2000, step: float = 0.05, goal_sample_rate: float = 0.2) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    pts = [start]
    parents = [-1]
    low, high = bounds

    def collision_free(p0, p1):
        return not segment_intersects_aabb(p0, p1, aabb_min, aabb_max)

    for it in range(max_iters):
        if rng.random() < goal_sample_rate:
            sample = goal
        else:
            sample = low + rng.random(3) * (high - low)
        # nearest
        dists = [np.linalg.norm(p - sample) for p in pts]
        idx = int(np.argmin(dists))
        direction = sample - pts[idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        new_pt = pts[idx] + direction / norm * step
        if collision_free(pts[idx], new_pt):
            parents.append(idx)
            pts.append(new_pt)
            if np.linalg.norm(new_pt - goal) < step and collision_free(new_pt, goal):
                # reconstruct
                path = [goal, new_pt]
                cur = len(pts) - 1
                while parents[cur] != -1:
                    cur = parents[cur]
                    path.append(pts[cur])
                return list(reversed(path))
    return []