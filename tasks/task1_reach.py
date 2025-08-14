import math
from typing import Tuple, Dict, Any

import numpy as np
import pybullet as p

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda, ik_move, step_simulation
from sim.metrics import l2_distance, write_results_json


def run(output_path: str = "outputs/task1_reach.mp4") -> Dict[str, Any]:
    client_id = setup_simulation(gui=False)

    # Load robot
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id, base_pos=(0, 0, 0))

    # Camera setup
    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Target: reachable point above the plane
    target = np.array([0.5, 0.0, 0.25])
    target_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_sphere, basePosition=target.tolist())

    # Home pose
    ik_move(robot_id, ee_idx, (0.4, 0.0, 0.4), None, arm_joint_indices, steps=180, client_id=client_id)

    # Move to target in small increments without orientation constraint
    num_steps = 300
    start = np.array([0.4, 0.0, 0.4])
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        pos = (1 - alpha) * start + alpha * target
        ik_move(robot_id, ee_idx, pos.tolist(), None, arm_joint_indices, steps=2, client_id=client_id)
        if i % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Converge with smaller step windows
    for _ in range(60):
        ik_move(robot_id, ee_idx, target.tolist(), None, arm_joint_indices, steps=1, client_id=client_id)
        recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Metrics: end-effector distance to target
    ee_pos = p.getLinkState(robot_id, ee_idx)[0]
    dist = l2_distance(tuple(ee_pos), tuple(target.tolist()))
    success = dist < 0.10
    metrics: Dict[str, Any] = {
        "task": "task1_reach",
        "success": bool(success),
        "distance_to_target": float(dist),
        "threshold": 0.10,
        "output_video": output_path,
    }
    write_results_json("outputs/results_task1.json", metrics)

    recorder.close()
    p.disconnect(client_id)
    return metrics


if __name__ == "__main__":
    run()