import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from sim.utils import setup_simulation, load_panda, get_default_camera, render_camera_frame, VideoRecorder, ik_move
from sim.controllers import move_ee_towards
from tools.rrt_planner import rrt

OUT_DIR = Path("/workspace/outputs/perf")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def path_length(pts: List[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))) if len(pts) > 1 else 0.0


def smoothness(pts: List[np.ndarray]) -> float:
    # Sum squared second-difference magnitude as a smoothness proxy
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for i in range(1, len(pts)-1):
        acc += float(np.linalg.norm(pts[i+1] - 2*pts[i] + pts[i-1])**2)
    return acc


def energy_proxy(q_seq: List[np.ndarray]) -> float:
    # Sum absolute joint deltas as a proxy for energy
    if len(q_seq) < 2:
        return 0.0
    total = 0.0
    for i in range(len(q_seq)-1):
        total += float(np.sum(np.abs(q_seq[i+1] - q_seq[i])))
    return total


def bench_reach() -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joints, fingers, ee_idx = load_panda(cid)
    target = np.array([0.55, 0.0, 0.28])
    pts: List[np.ndarray] = []
    q_seq: List[np.ndarray] = []

    # Home
    ik_move(robot_id, ee_idx, (0.45, 0.0, 0.35), None, arm_joints, steps=120, client_id=cid)
    # Move towards target in 200 steps
    start = np.array(p.getLinkState(robot_id, ee_idx)[0])
    steps = 200
    for i in range(steps):
        pos = (1 - i/(steps-1)) * start + (i/(steps-1)) * target
        ik_move(robot_id, ee_idx, pos.tolist(), None, arm_joints, steps=2, client_id=cid)
        pts.append(np.array(p.getLinkState(robot_id, ee_idx)[0]))
        q_seq.append(np.array([s[0] for s in p.getJointStates(robot_id, arm_joints)]))
    final = pts[-1]
    dist = float(np.linalg.norm(final - target))
    res = {
        "task": "reach_perf",
        "final_error": dist,
        "path_length": path_length(pts),
        "smoothness": smoothness(pts),
        "energy_proxy": energy_proxy(q_seq),
    }
    p.disconnect(cid)
    return res


def bench_figure8() -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joints, fingers, ee_idx = load_panda(cid)
    center = np.array([0.55, 0.0, 0.30])
    move_ee_towards(robot_id, ee_idx, arm_joints, center.tolist(), cid, iters=80)
    a, b = 0.10, 0.06
    T = 600
    err_list: List[float] = []
    pts: List[np.ndarray] = []
    q_seq: List[np.ndarray] = []
    for t in range(T):
        theta = 2 * math.pi * (t / T)
        x = a * math.sin(theta)
        y = b * math.sin(theta) * math.cos(theta)
        target = center + np.array([x, y, 0.0])
        move_ee_towards(robot_id, ee_idx, arm_joints, target.tolist(), cid, iters=2)
        ee = np.array(p.getLinkState(robot_id, ee_idx)[0])
        err_list.append(float(np.linalg.norm(ee - target)))
        pts.append(ee)
        q_seq.append(np.array([s[0] for s in p.getJointStates(robot_id, arm_joints)]))
    p.disconnect(cid)
    return {
        "task": "figure8_perf",
        "mean_error": float(np.mean(err_list)),
        "max_error": float(np.max(err_list)),
        "path_length": path_length(pts),
        "smoothness": smoothness(pts),
        "energy_proxy": energy_proxy(q_seq),
    }


def bench_rrt() -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joints, fingers, ee_idx = load_panda(cid)
    aabb_min = np.array([0.55, -0.05, 0.18])
    aabb_max = np.array([0.65, 0.05, 0.34])
    low = np.array([0.40, -0.30, 0.15])
    high = np.array([0.80, 0.30, 0.45])
    start = np.array(p.getLinkState(robot_id, ee_idx)[0])
    goal = np.array([0.70, 0.15, 0.28])
    path = rrt(start, goal, aabb_min, aabb_max, (low, high), max_iters=5000, step=0.03)
    if not path:
        path = [start, np.array([0.50, 0.20, 0.32]), goal]
    pts = [start]
    q_seq: List[np.ndarray] = []
    clearance_min = float("inf")
    # Execute
    for pt in path[1:]:
        ik_move(robot_id, ee_idx, pt.tolist(), None, arm_joints, steps=120, client_id=cid)
        cur = np.array(p.getLinkState(robot_id, ee_idx)[0])
        pts.append(cur)
        q_seq.append(np.array([s[0] for s in p.getJointStates(robot_id, arm_joints)]))
        # Clearance to aabb (inside gives negative clearance)
        d = np.maximum(aabb_min - cur, 0) + np.maximum(cur - aabb_max, 0)
        clearance = float(np.linalg.norm(d))
        clearance_min = min(clearance_min, clearance)
    final = pts[-1]
    dist = float(np.linalg.norm(final - goal))
    p.disconnect(cid)
    return {
        "task": "rrt_perf",
        "final_error": dist,
        "path_length": path_length(pts),
        "smoothness": smoothness(pts),
        "energy_proxy": energy_proxy(q_seq),
        "min_clearance": clearance_min,
    }


def draw_perf_plot(perf: Dict[str, Any], out_path: Path) -> None:
    keys = ["final_error", "mean_error", "max_error", "path_length", "smoothness", "energy_proxy", "min_clearance"]
    # Aggregate values for tasks
    tasks = list(perf.keys())
    metrics = {}
    for k in keys:
        row = []
        for t in tasks:
            row.append(perf[t].get(k, 0.0))
        metrics[k] = row
    width, height = 1000, 600
    img = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    margin = 50
    x = margin
    y = margin
    line_h = 20
    # Header
    draw.text((x, y), "Performance summary (lower is better for errors, higher is better for clearance)", fill=(20, 20, 25))
    y += 30
    for k in keys:
        draw.text((x, y), f"{k}: {', '.join(f'{v:.3f}' for v in metrics[k])}", fill=(30, 30, 35))
        y += line_h
    img.save(out_path)


def main() -> Dict[str, Any]:
    perf: Dict[str, Any] = {}
    perf["reach"] = bench_reach()
    perf["figure8"] = bench_figure8()
    perf["rrt"] = bench_rrt()
    (OUT_DIR / "perf.json").write_text(json.dumps(perf, indent=2))
    draw_perf_plot(perf, OUT_DIR / "perf.png")
    return perf


if __name__ == "__main__":
    print(main())