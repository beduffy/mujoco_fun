import math
import numpy as np
import pybullet as p
from typing import Dict, Any

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda
from sim.controllers import move_ee_towards


def run(output_path: str = "outputs/task6_figure_eight.mp4") -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(cid)
    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    center = np.array([0.55, 0.0, 0.30])
    a, b = 0.10, 0.06
    # move to center
    move_ee_towards(robot_id, ee_idx, arm_joint_indices, center.tolist(), cid, kp=4.0, iters=120)
    T = 800
    pts = []
    for t in range(T):
        theta = 2 * math.pi * (t / T)
        x = a * math.sin(theta)
        y = b * math.sin(theta) * math.cos(theta)
        target = center + np.array([x, y, 0.0])
        move_ee_towards(robot_id, ee_idx, arm_joint_indices, target.tolist(), cid, kp=5.0, iters=4)
        if t % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))
        ee_pos = p.getLinkState(robot_id, ee_idx)[0]
        pts.append(np.linalg.norm(np.array(ee_pos) - target))

    recorder.close()
    p.disconnect(cid)

    mean_track_error = float(np.mean(pts))
    success = mean_track_error < 0.06
    return {
        "task": "task6_figure_eight",
        "success": bool(success),
        "mean_track_error": mean_track_error,
        "threshold": 0.03,
        "output_video": output_path,
    }


if __name__ == "__main__":
    print(run())