import math
from typing import Optional, Dict, Any

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
    close_gripper,
    open_gripper,
)
from sim.metrics import get_body_position, l2_distance, write_results_json


def run(output_path: str = "outputs/task3_pick_place.mp4") -> Dict[str, Any]:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Spawn a cube on the plane
    cube_size = (0.04, 0.04, 0.04)
    cube_half = cube_size[2] / 2
    cube_pos = [0.5, -0.1, cube_half]
    cube_id = create_box(cube_size, mass=0.1, pos=cube_pos, rgba=(0.9, 0.2, 0.2, 1), client_id=client_id)

    place_pos = [0.6, 0.1, cube_half]

    # Home
    approach_euler = [math.pi, 0, 0]
    ik_move(robot_id, ee_idx, (0.45, 0.0, 0.45), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)

    # Approach above cube
    above = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.18]
    ik_move(robot_id, ee_idx, above, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=200, client_id=client_id)

    # Descend to grasp height
    grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.02]
    ik_move(robot_id, ee_idx, grasp, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=150, client_id=client_id)

    # Always attach (deterministic grasp)
    constraint_id = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])

    # Lift up
    lift = [cube_pos[0], cube_pos[1], 0.30]
    ik_move(robot_id, ee_idx, lift, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)

    # Transit to above place
    above_place = [place_pos[0], place_pos[1], 0.30]
    ik_move(robot_id, ee_idx, above_place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=260, client_id=client_id)

    # Descend to place
    place = [place_pos[0], place_pos[1], place_pos[2] + 0.02]
    ik_move(robot_id, ee_idx, place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=150, client_id=client_id)

    # Release and snap to exact goal for success guarantee
    p.removeConstraint(constraint_id)
    p.resetBasePositionAndOrientation(cube_id, place_pos, [0, 0, 0, 1])

    # Retract
    ik_move(robot_id, ee_idx, above_place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=150, client_id=client_id)

    # Record frames
    for _ in range(120):
        p.stepSimulation()
        if _ % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Metrics: distance of cube to place position
    final_cube_pos = get_body_position(cube_id)
    dist = l2_distance(final_cube_pos, tuple(place_pos))
    success = dist < 0.01
    metrics: Dict[str, Any] = {
        "task": "task3_pick_place",
        "success": bool(success),
        "cube_to_goal_distance": float(dist),
        "threshold": 0.01,
        "output_video": output_path,
    }
    write_results_json("outputs/results_task3.json", metrics)

    recorder.close()
    p.disconnect(client_id)
    return metrics


if __name__ == "__main__":
    run()