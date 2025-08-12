import math
from typing import List

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


def run(output_path: str = "outputs/task4_obstacle_pick_place.mp4") -> None:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Objects
    cube_size = (0.04, 0.04, 0.04)
    cube_pos = [0.45, -0.15, 0.02]
    cube_id = create_box(cube_size, mass=0.1, pos=cube_pos, rgba=(0.2, 0.6, 0.9, 1), client_id=client_id)

    obstacle_size = (0.12, 0.12, 0.12)
    obstacle_pos = [0.55, -0.05, 0.06]
    obstacle_id = create_box(obstacle_size, mass=0, pos=obstacle_pos, rgba=(0.2, 0.8, 0.2, 1), client_id=client_id)

    place_pos = [0.65, 0.2, 0.02]

    approach_euler = [math.pi, 0, 0]

    # Sequence of via points to avoid obstacle
    via_points: List[List[float]] = [
        [cube_pos[0], cube_pos[1], 0.28],
        [cube_pos[0], cube_pos[1], cube_pos[2] + 0.02],
        [cube_pos[0], cube_pos[1], 0.28],
        [0.50, 0.10, 0.28],  # go around obstacle on Y+
        [place_pos[0], place_pos[1], 0.28],
        [place_pos[0], place_pos[1], place_pos[2] + 0.02],
        [place_pos[0], place_pos[1], 0.28],
    ]

    # Move to home
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.35), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # Create a fixed constraint when descending to grasp
    constraint_id = -1

    for idx, waypoint in enumerate(via_points):
        ik_move(robot_id, ee_idx, waypoint, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)
        # If at the grasp waypoint, attach cube if close
        if idx == 1:
            ee_state = p.getLinkState(robot_id, ee_idx)
            ee_pos = np.array(ee_state[0])
            if np.linalg.norm(ee_pos - np.array(cube_pos)) < 0.03:
                constraint_id = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        # If at the place waypoint, detach
        if idx == 5 and constraint_id != -1:
            p.removeConstraint(constraint_id)

        # Record frame every few sim steps
        for _ in range(60):
            p.stepSimulation()
            if _ % 3 == 0:
                recorder.add_frame(render_camera_frame(view, proj, w, h))

    recorder.close()
    p.disconnect(client_id)


if __name__ == "__main__":
    run()