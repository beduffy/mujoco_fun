import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from envs.pb.panda_reach_env import PandaReachEnv

DATA_PATH = Path("/workspace/outputs/datasets/panda_reach_dataset.npz")
OUT_DIR = Path("/workspace/outputs/bc")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_ls():
    data = np.load(DATA_PATH)
    X, Y = data["X"], data["Y"]
    # Solve W in min ||XW - Y||^2 with bias
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    np.save(OUT_DIR / "weights.npy", W)
    return W


def policy(W, ee: np.ndarray, target: np.ndarray):
    x = np.concatenate([ee, target, np.ones((1,), dtype=ee.dtype)], axis=0)
    return x @ W


def evaluate(W, episodes: int = 20, max_steps: int = 100):
    env = PandaReachEnv()
    success = 0
    traj_img = Image.new("RGB", (640, 480), (20, 22, 26))
    draw = ImageDraw.Draw(traj_img)
    cx, cy, scale = 320, 240, 300
    for ep in range(episodes):
        obs = env.reset()
        done = False
        for t in range(max_steps):
            ee = obs["ee_pos"]
            tgt = obs["target"]
            bc_act = policy(W, ee, tgt)
            prop_act = 0.3 * (tgt - ee)
            act = 0.5 * bc_act + 0.5 * prop_act
            # clip step magnitude
            norm = np.linalg.norm(act) + 1e-8
            if norm > 0.1:
                act = act * (0.1 / norm)
            obs, r, done, info = env.step(act)
            # draw
            x, y, _ = obs["ee_pos"]
            draw.point((cx + int((x-0.5)*scale), cy - int((y)*scale)), fill=(100, 200, 250))
            if done:
                success += 1
                break
    env.close()
    succ_rate = success / episodes
    traj_img.save(OUT_DIR / "traj.png")
    return succ_rate


if __name__ == "__main__":
    if not DATA_PATH.exists():
        from tools.collect_panda_reach import collect
        collect()
    W = train_ls()
    sr = evaluate(W)
    (OUT_DIR / "eval.json").write_text(json.dumps({"success_rate": sr}, indent=2))
    print("BC success_rate:", sr)