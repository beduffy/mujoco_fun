import math
from typing import Tuple

import numpy as np
import pybullet as p

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda, ik_move, step_simulation


def run(output_path: str = "outputs/task2_path_tracing.mp4") -> None:
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
    for f in range(total_frames):
        theta = 2 * math.pi * (f / total_frames)
        pos = center + radius * np.array([math.cos(theta), math.sin(theta), 0.0])
        ik_move(robot_id, ee_idx, pos.tolist(), p.getQuaternionFromEuler(normal_euler), arm_joint_indices, steps=1, client_id=client_id)
        if f % 2 == 0:
            rgb = render_camera_frame(view, proj, w, h)
            recorder.add_frame(rgb)

    # Hold a bit
    for _ in range(30):
        step_simulation(1, client_id)
        rgb = render_camera_frame(view, proj, w, h)
        recorder.add_frame(rgb)

    recorder.close()
    p.disconnect(client_id)


if __name__ == "__main__":
    run()