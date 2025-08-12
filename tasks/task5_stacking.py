import math
from typing import Tuple

import numpy as np
import pybullet as p

from sim.utils import (
    setup_simulation,
    get_default_camera,
    render_camera_frame,
    VideoRecorder,
    load_panda,
    ik_move,
    step_simulation,
    create_box,
)


def pick_with_suction(robot_id: int, ee_idx: int, target_pos: Tuple[float, float, float]) -> int:
    ee_state = p.getLinkState(robot_id, ee_idx)
    ee_pos = np.array(ee_state[0])
    if np.linalg.norm(ee_pos - np.array(target_pos)) < 0.03:
        return p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=-1, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
    return -1


def run(output_path: str = "outputs/task5_stacking.mp4") -> None:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    size = (0.04, 0.04, 0.04)
    a_pos = [0.45, -0.10, 0.02]
    b_pos = [0.55, -0.10, 0.02]
    goal_pos = [0.60, 0.10, 0.02]

    cube_a = create_box(size, mass=0.1, pos=a_pos, rgba=(0.9, 0.4, 0.2, 1), client_id=client_id)
    cube_b = create_box(size, mass=0.1, pos=b_pos, rgba=(0.2, 0.4, 0.9, 1), client_id=client_id)

    approach_euler = [math.pi, 0, 0]

    def grasp_and_place(cube_id: int, source_xy: Tuple[float, float], z: float) -> None:
        above = [source_xy[0], source_xy[1], 0.28]
        down = [source_xy[0], source_xy[1], z + 0.02]
        ik_move(robot_id, ee_idx, above, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)
        ik_move(robot_id, ee_idx, down, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=150, client_id=client_id)
        # Attach
        ee_state = p.getLinkState(robot_id, ee_idx)
        ee_pos = np.array(ee_state[0])
        constraint_id = -1
        if np.linalg.norm(ee_pos - np.array([source_xy[0], source_xy[1], z])) < 0.04:
            constraint_id = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        lift = [source_xy[0], source_xy[1], 0.28]
        ik_move(robot_id, ee_idx, lift, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)
        above_goal = [goal_pos[0], goal_pos[1], 0.28]
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=240, client_id=client_id)
        down_goal = [goal_pos[0], goal_pos[1], 0.02 + z]
        ik_move(robot_id, ee_idx, down_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=150, client_id=client_id)
        if constraint_id != -1:
            p.removeConstraint(constraint_id)
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # Move to home
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.35), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # First cube to base
    grasp_and_place(cube_a, (a_pos[0], a_pos[1]), z=0.02)

    # Second cube to stack on top
    grasp_and_place(cube_b, (b_pos[0], b_pos[1]), z=0.06)  # place on top (base 0.02 + cube half 0.02 + clearance 0.02)

    # Record a bit
    for _ in range(240):
        p.stepSimulation()
        if _ % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))

    recorder.close()
    p.disconnect(client_id)


if __name__ == "__main__":
    run()