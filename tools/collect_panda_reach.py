import os, json
import numpy as np
from pathlib import Path
from envs.pb.panda_reach_env import PandaReachEnv

OUT_DIR = Path("/workspace/outputs/datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect(num_episodes: int = 20, max_steps: int = 100, alpha: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    env = PandaReachEnv()
    X_list, Y_list = [], []
    success = 0
    for ep in range(num_episodes):
        # randomize target per episode
        target = np.array([0.45, rng.uniform(-0.15, 0.15), rng.uniform(0.2, 0.35)], dtype=np.float32)
        env.target = target
        obs = env.reset()
        done = False
        for t in range(max_steps):
            ee = obs["ee_pos"]
            tgt = obs["target"]
            action = (tgt - ee) * alpha
            feat = np.concatenate([ee, tgt], axis=0)
            X_list.append(feat)
            Y_list.append(action)
            obs, r, done, info = env.step(action)
            if done:
                success += 1
                break
    env.close()

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    out_npz = OUT_DIR / "panda_reach_dataset.npz"
    np.savez_compressed(out_npz, X=X, Y=Y)
    summary = {"episodes": num_episodes, "successes": int(success), "dataset": str(out_npz)}
    (OUT_DIR / "panda_reach_dataset.json").write_text(json.dumps(summary, indent=2))
    print("Saved:", out_npz, "summary:", summary)


if __name__ == "__main__":
    collect()