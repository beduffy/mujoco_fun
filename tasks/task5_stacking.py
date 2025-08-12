import math
from typing import Tuple, Dict, Any

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
from sim.metrics import get_body_position, l2_distance, write_results_json


def pick_with_suction(robot_id: int, ee_idx: int, target_pos: Tuple[float, float, float]) -> int:
    ee_state = p.getLinkState(robot_id, ee_idx)
    ee_pos = np.array(ee_state[0])
    if np.linalg.norm(ee_pos - np.array(target_pos)) < 0.05:
        return p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=-1, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
    return -1


def run(output_path: str = "outputs/task5_stacking.mp4") -> Dict[str, Any]:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    size = (0.04, 0.04, 0.04)
    half = size[2] / 2
    a_pos = [0.45, -0.10, half]
    b_pos = [0.55, -0.10, half]
    goal_pos = [0.60, 0.10, half]

    cube_a = create_box(size, mass=0.1, pos=a_pos, rgba=(0.9, 0.4, 0.2, 1), client_id=client_id)
    cube_b = create_box(size, mass=0.1, pos=b_pos, rgba=(0.2, 0.4, 0.9, 1), client_id=client_id)

    approach_euler = [math.pi, 0, 0]

    def grasp_and_place(cube_id: int, source_xy: Tuple[float, float], z: float) -> None:
        above = [source_xy[0], source_xy[1], 0.32]
        down = [source_xy[0], source_xy[1], z + 0.02]
        ik_move(robot_id, ee_idx, above, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=200, client_id=client_id)
        ik_move(robot_id, ee_idx, down, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)
        # Attach
        ee_state = p.getLinkState(robot_id, ee_idx)
        ee_pos = np.array(ee_state[0])
        constraint_id = -1
        if np.linalg.norm(ee_pos - np.array([source_xy[0], source_xy[1], z])) < 0.05:
            constraint_id = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        lift = [source_xy[0], source_xy[1], 0.32]
        ik_move(robot_id, ee_idx, lift, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=160, client_id=client_id)
        above_goal = [goal_pos[0], goal_pos[1], 0.32]
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=280, client_id=client_id)
        down_goal = [goal_pos[0], goal_pos[1], z + 0.02]
        ik_move(robot_id, ee_idx, down_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)
        if constraint_id != -1:
            p.removeConstraint(constraint_id)
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=160, client_id=client_id)

    # Move to home
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.36), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)

    # First cube to base
    grasp_and_place(cube_a, (a_pos[0], a_pos[1]), z=goal_pos[2])

    # Second cube to stack on top
    grasp_and_place(cube_b, (b_pos[0], b_pos[1]), z=goal_pos[2] + size[2])

    # Record a bit
    for _ in range(240):
        p.stepSimulation()
        if _ % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Metrics: both cubes near their expected stacked positions
    goal_a = tuple(goal_pos)
    goal_b = (goal_pos[0], goal_pos[1], goal_pos[2] + size[2])
    final_a = get_body_position(cube_a)
    final_b = get_body_position(cube_b)
    dist_a = l2_distance(final_a, goal_a)
    dist_b = l2_distance(final_b, goal_b)
    success = (dist_a < 0.05) and (dist_b < 0.06)
    metrics: Dict[str, Any] = {
        "task": "task5_stacking",
        "success": bool(success),
        "cube_a_distance": float(dist_a),
        "cube_b_distance": float(dist_b),
        "thresholds": {"a": 0.05, "b": 0.06},
        "output_video": output_path,
    }
    write_results_json("outputs/results_task5.json", metrics)

    recorder.close()
    p.disconnect(client_id)
    return metrics


if __name__ == "__main__":
    run()