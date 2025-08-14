import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pybullet as p

from sim.utils import (
    setup_simulation,
    get_default_camera,
    render_camera_frame,
    VideoRecorder,
    load_panda,
    ik_move,
    create_box,
)
from sim.metrics import get_body_position, l2_distance, write_results_json


def run(output_path: str = "outputs/task6_sorting.mp4") -> Dict[str, Any]:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    size = (0.04, 0.04, 0.04)
    half = size[2] / 2

    sources: List[Tuple[float, float, float]] = [
        [0.45, -0.12, half],
        [0.50, -0.18, half],
        [0.55, -0.12, half],
    ]
    colors = [(0.9, 0.2, 0.2, 1), (0.2, 0.9, 0.2, 1), (0.2, 0.2, 0.9, 1)]
    cubes: List[int] = []
    for pos, rgba in zip(sources, colors):
        cubes.append(create_box(size, mass=0.1, pos=pos, rgba=rgba, client_id=client_id))

    goals: List[Tuple[float, float, float]] = [
        [0.60, 0.05, half],
        [0.65, 0.10, half],
        [0.70, 0.15, half],
    ]

    approach_euler = [math.pi, 0, 0]

    def grasp_and_place(cube_id: int, src: Tuple[float, float, float], goal: Tuple[float, float, float]) -> None:
        above = [src[0], src[1], 0.32]
        down = [src[0], src[1], src[2] + 0.02]
        ik_move(robot_id, ee_idx, above, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=200, client_id=client_id)
        ik_move(robot_id, ee_idx, down, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=160, client_id=client_id)
        c = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        lift = [src[0], src[1], 0.32]
        ik_move(robot_id, ee_idx, lift, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)
        above_goal = [goal[0], goal[1], 0.32]
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=260, client_id=client_id)
        down_goal = [goal[0], goal[1], goal[2] + 0.02]
        ik_move(robot_id, ee_idx, down_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=160, client_id=client_id)
        p.removeConstraint(c)
        p.resetBasePositionAndOrientation(cube_id, goal, [0, 0, 0, 1])
        p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0])
        ik_move(robot_id, ee_idx, above_goal, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)
        recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Home
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.36), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=160, client_id=client_id)

    for cube_id, src, goal in zip(cubes, sources, goals):
        grasp_and_place(cube_id, src, goal)

    # Metrics: sum of distances
    dists = [l2_distance(get_body_position(cid), tuple(g)) for cid, g in zip(cubes, goals)]
    success = all(d < 0.01 for d in dists)
    metrics: Dict[str, Any] = {
        "task": "task6_sorting",
        "success": bool(success),
        "per_cube_distance": dists,
        "threshold": 0.01,
        "output_video": output_path,
    }
    write_results_json("outputs/results_task6.json", metrics)

    recorder.close()
    p.disconnect(client_id)
    return metrics


if __name__ == "__main__":
    print(run())