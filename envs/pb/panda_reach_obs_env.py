import numpy as np
from typing import Dict, Any, Tuple
import pybullet as p

from sim.utils import setup_simulation, load_panda, ik_move, render_camera_frame, get_default_camera, create_box


class PandaReachObstacleEnv:
    def __init__(self, target: Tuple[float, float, float] = (0.65, 0.0, 0.25),
                 obstacle_center: Tuple[float, float, float] = (0.55, 0.0, 0.20),
                 obstacle_size: Tuple[float, float, float] = (0.12, 0.20, 0.20)):
        self.target = np.array(target, dtype=np.float32)
        self.client_id = None
        self.robot_id = None
        self.arm_indices = None
        self.ee_idx = None
        self.view = None
        self.proj = None
        self.w = 0
        self.h = 0
        self.obstacle_id = None
        self.obstacle_aabb = None  # (min, max)
        self.obstacle_center = np.array(obstacle_center, dtype=np.float32)
        self.obstacle_size = np.array(obstacle_size, dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        if self.client_id is not None:
            p.disconnect(self.client_id)
        self.client_id = setup_simulation(gui=False)
        self.robot_id, self.arm_indices, _, self.ee_idx = load_panda(self.client_id)
        self.view, self.proj, self.w, self.h = get_default_camera(320, 240)
        # create obstacle box
        size = tuple(self.obstacle_size.tolist())
        pos = tuple(self.obstacle_center.tolist())
        self.obstacle_id = create_box(size, mass=0.0, pos=pos, rgba=(0.2, 0.7, 0.2, 1), client_id=self.client_id)
        aabb_min = self.obstacle_center - self.obstacle_size / 2
        aabb_max = self.obstacle_center + self.obstacle_size / 2
        self.obstacle_aabb = (aabb_min, aabb_max)
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        return {
            "ee_pos": np.array(ee_pos, dtype=np.float32),
            "target": self.target.copy(),
            "obstacle": {
                "center": self.obstacle_center.copy(),
                "size": self.obstacle_size.copy(),
            },
        }

    @staticmethod
    def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        # Liang-Barsky clipping in 3D via slab method
        tmin, tmax = 0.0, 1.0
        d = p1 - p0
        for i in range(3):
            if abs(d[i]) < 1e-9:
                if p0[i] < aabb_min[i] or p0[i] > aabb_max[i]:
                    return False  # parallel and outside slab
            else:
                ood = 1.0 / d[i]
                t1 = (aabb_min[i] - p0[i]) * ood
                t2 = (aabb_max[i] - p0[i]) * ood
                t1, t2 = (t1, t2) if t1 <= t2 else (t2, t1)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_idx)[0])
        tgt = ee_pos + np.array(action, dtype=np.float32)
        # collision check on segment
        aabb_min, aabb_max = self.obstacle_aabb
        collides = self._segment_intersects_aabb(ee_pos, tgt, aabb_min, aabb_max)
        if not collides:
            ik_move(self.robot_id, self.ee_idx, tgt.tolist(), None, self.arm_indices, steps=2, client_id=self.client_id)
        obs = self._get_obs()
        dist = float(np.linalg.norm(obs["ee_pos"] - obs["target"]))
        reward = -dist - (0.5 if collides else 0.0)
        done = dist < 0.03
        info = {"dist": dist, "collides": bool(collides)}
        return obs, reward, done, info

    def render(self) -> np.ndarray:
        return render_camera_frame(self.view, self.proj, self.w, self.h)

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None