import numpy as np
from typing import Dict, Any, Tuple
import pybullet as p

from sim.utils import setup_simulation, load_panda, ik_move, render_camera_frame, get_default_camera


class PandaReachEnv:
    def __init__(self, target: Tuple[float, float, float] = (0.5, 0.0, 0.25)):
        self.target = np.array(target, dtype=np.float32)
        self.client_id = None
        self.robot_id = None
        self.arm_indices = None
        self.ee_idx = None
        self.view = None
        self.proj = None
        self.w = 0
        self.h = 0

    def reset(self) -> Dict[str, Any]:
        if self.client_id is not None:
            p.disconnect(self.client_id)
        self.client_id = setup_simulation(gui=False)
        self.robot_id, self.arm_indices, _, self.ee_idx = load_panda(self.client_id)
        self.view, self.proj, self.w, self.h = get_default_camera(320, 240)
        obs = self._get_obs()
        return obs

    def _get_obs(self) -> Dict[str, Any]:
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        return {
            "ee_pos": np.array(ee_pos, dtype=np.float32),
            "target": self.target.copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # action is target delta for EE
        ee_pos = p.getLinkState(self.robot_id, self.ee_idx)[0]
        tgt = np.array(ee_pos) + np.array(action, dtype=np.float32)
        ik_move(self.robot_id, self.ee_idx, tgt.tolist(), None, self.arm_indices, steps=2, client_id=self.client_id)
        obs = self._get_obs()
        dist = float(np.linalg.norm(obs["ee_pos"] - obs["target"]))
        reward = -dist
        done = dist < 0.03
        info = {"dist": dist}
        return obs, reward, done, info

    def render(self) -> np.ndarray:
        return render_camera_frame(self.view, self.proj, self.w, self.h)

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None