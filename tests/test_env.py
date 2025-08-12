from envs.pb.panda_reach_env import PandaReachEnv
import numpy as np

def test_env_smoke():
    env = PandaReachEnv()
    obs = env.reset()
    for _ in range(200):
        ee = obs["ee_pos"]
        tgt = obs["target"]
        delta = (tgt - ee) * 0.2
        obs, r, done, info = env.step(delta)
        if done:
            break
    env.close()
    assert done, f"env did not finish: dist={info['dist']:.3f}"