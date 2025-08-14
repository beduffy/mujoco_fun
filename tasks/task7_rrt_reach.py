import numpy as np
import pybullet as p
from typing import Dict, Any

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda, ik_move, create_box
from tools.rrt_planner import rrt
from tools.rrt_planner import shortcut


def run(output_path: str = "outputs/task7_rrt_reach.mp4") -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(cid)
    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Define obstacle AABB and visualize
    aabb_min = np.array([0.55, -0.05, 0.18])
    aabb_max = np.array([0.65, 0.05, 0.34])
    size = (aabb_max - aabb_min)
    center = (aabb_min + aabb_max) / 2
    create_box(tuple(size.tolist()), mass=0.0, pos=tuple(center.tolist()), rgba=(0.2, 0.8, 0.2, 1), client_id=cid)

    # Start and goal
    ee_pos = np.array(p.getLinkState(robot_id, ee_idx)[0])
    goal = np.array([0.70, 0.15, 0.28])

    # Plan with RRT
    low = np.array([0.40, -0.30, 0.15])
    high = np.array([0.80, 0.30, 0.45])
    path = rrt(ee_pos, goal, aabb_min, aabb_max, (low, high), max_iters=5000, step=0.03)
    if path:
        path = shortcut(path, aabb_min, aabb_max)

    # If planning failed, fall back to via point around obstacle
    if not path:
      path = [ee_pos, np.array([0.50, 0.20, 0.32]), goal]

    # Execute path
    for pt in path:
        ik_move(robot_id, ee_idx, pt.tolist(), None, arm_joint_indices, steps=120, client_id=cid)
        recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Hold final
    for _ in range(30):
        p.stepSimulation(cid)
        recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Metric
    final = np.array(p.getLinkState(robot_id, ee_idx)[0])
    dist = float(np.linalg.norm(final - goal))
    success = dist < 0.04

    recorder.close()
    p.disconnect(cid)

    return {
        "task": "task7_rrt_reach",
        "success": bool(success),
        "ee_goal_distance": dist,
        "threshold": 0.04,
        "output_video": output_path,
    }


if __name__ == "__main__":
    print(run())