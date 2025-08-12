import math
from typing import Tuple

import numpy as np
import pybullet as p

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda, ik_move, step_simulation


def run(output_path: str = "outputs/task1_reach.mp4") -> None:
    client_id = setup_simulation(gui=False)

    # Load robot
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id, base_pos=(0, 0, 0))

    # Camera setup
    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Target: a reachable point above the table
    target = np.array([0.5, 0.0, 0.1])
    target_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_sphere, basePosition=target.tolist())

    # Home pose
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.4), p.getQuaternionFromEuler([math.pi, 0, 0]), arm_joint_indices, steps=120, client_id=client_id)

    # Record initial frame
    rgb = render_camera_frame(view, proj, w, h)
    recorder.add_frame(rgb)

    # Move to target in small increments
    num_steps = 240
    start = np.array([0.5, 0.0, 0.4])
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        pos = (1 - alpha) * start + alpha * target
        ik_move(robot_id, ee_idx, pos.tolist(), p.getQuaternionFromEuler([math.pi, 0, 0]), arm_joint_indices, steps=2, client_id=client_id)
        if i % 2 == 0:
            rgb = render_camera_frame(view, proj, w, h)
            recorder.add_frame(rgb)

    # Hold for a moment
    for _ in range(30):
        step_simulation(1, client_id)
        rgb = render_camera_frame(view, proj, w, h)
        recorder.add_frame(rgb)

    recorder.close()
    p.disconnect(client_id)


if __name__ == "__main__":
    run()