import math
from typing import Tuple, Dict, Any, List

import numpy as np
import pybullet as p

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda, ik_move, step_simulation
from sim.metrics import path_radial_error_and_coverage, write_results_json


def run(output_path: str = "outputs/task2_path_tracing.mp4") -> Dict[str, Any]:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    center = np.array([0.5, 0.0, 0.25])
    radius = 0.10
    normal_euler = [math.pi, 0, 0]

    # Move to start
    start = center + np.array([radius, 0, 0])
    ik_move(robot_id, ee_idx, start.tolist(), p.getQuaternionFromEuler(normal_euler), arm_joint_indices, steps=120, client_id=client_id)

    total_frames = 600
    ee_path: List[tuple] = []
    for f in range(total_frames):
        theta = 2 * math.pi * (f / total_frames)
        pos = center + radius * np.array([math.cos(theta), math.sin(theta), 0.0])
        ik_move(robot_id, ee_idx, pos.tolist(), p.getQuaternionFromEuler(normal_euler), arm_joint_indices, steps=1, client_id=client_id)
        if f % 2 == 0:
            rgb = render_camera_frame(view, proj, w, h)
            recorder.add_frame(rgb)
        ee_pos = p.getLinkState(robot_id, ee_idx)[0]
        ee_path.append(tuple(ee_pos))

    # Hold a bit
    for _ in range(30):
        step_simulation(1, client_id)
        rgb = render_camera_frame(view, proj, w, h)
        recorder.add_frame(rgb)

    # Metrics: path radial error and coverage
    mean_err, coverage_deg = path_radial_error_and_coverage(ee_path, tuple(center.tolist()), radius)
    success = (mean_err < 0.02) and (coverage_deg > 320.0)
    metrics: Dict[str, Any] = {
        "task": "task2_path_tracing",
        "success": bool(success),
        "mean_radial_error": float(mean_err),
        "coverage_deg": float(coverage_deg),
        "error_threshold": 0.02,
        "coverage_threshold": 320.0,
        "output_video": output_path,
    }
    write_results_json("outputs/results_task2.json", metrics)

    recorder.close()
    p.disconnect(client_id)
    return metrics


if __name__ == "__main__":
    run()