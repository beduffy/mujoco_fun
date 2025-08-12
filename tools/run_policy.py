import numpy as np
from pathlib import Path
from typing import Literal

from envs.pb.panda_reach_env import PandaReachEnv
from sim.safety import rate_limit
from real.robot_interface import RobotInterface

WEIGHTS = Path("/workspace/outputs/bc/weights.npy")


def run(mode: Literal["sim", "ros2"] = "sim", steps: int = 200):
    W = np.load(WEIGHTS)
    if mode == "sim":
        env = PandaReachEnv()
        obs = env.reset()
        prev_act = np.zeros(3, dtype=np.float32)
        for _ in range(steps):
            ee, tgt = obs["ee_pos"], obs["target"]
            x = np.concatenate([ee, tgt, np.ones((1,), dtype=ee.dtype)], axis=0)
            act = x @ W
            act = rate_limit(prev_act, act, max_step=0.1)
            prev_act = act
            obs, r, done, info = env.step(act)
            if done:
                break
        env.close()
    else:
        robot = RobotInterface(mode="ros2")
        # Stub: read current joints, compute target joints from policy delta via Jacobian/IK, send using safety limits
        pass


if __name__ == "__main__":
    run("sim")