import math
from typing import Optional

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


def run(output_path: str = "outputs/task3_pick_place.mp4") -> None:
    client_id = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(client_id)

    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    # Spawn a cube on the table
    cube_size = (0.04, 0.04, 0.04)
    cube_pos = [0.5, -0.1, 0.02]
    cube_id = create_box(cube_size, mass=0.1, pos=cube_pos, rgba=(0.9, 0.2, 0.2, 1), client_id=client_id)

    place_pos = [0.6, 0.1, 0.02]

    # Home
    approach_euler = [math.pi, 0, 0]
    ik_move(robot_id, ee_idx, (0.5, 0.0, 0.4), p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    recorder.add_frame(render_camera_frame(view, proj, w, h))

    # Approach above cube
    above = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
    ik_move(robot_id, ee_idx, above, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)

    # Descend to grasp height
    grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.015]
    ik_move(robot_id, ee_idx, grasp, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # Simulate suction: create fixed constraint if close enough
    ee_state = p.getLinkState(robot_id, ee_idx)
    ee_pos = np.array(ee_state[0])
    if np.linalg.norm(ee_pos - np.array(cube_pos)) < 0.03:
        constraint_id = p.createConstraint(parentBodyUniqueId=robot_id, parentLinkIndex=ee_idx, childBodyUniqueId=cube_id, childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
    else:
        constraint_id = -1

    # Lift up
    lift = [cube_pos[0], cube_pos[1], 0.25]
    ik_move(robot_id, ee_idx, lift, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=180, client_id=client_id)

    # Transit to above place
    above_place = [place_pos[0], place_pos[1], 0.25]
    ik_move(robot_id, ee_idx, above_place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=240, client_id=client_id)

    # Descend to place
    place = [place_pos[0], place_pos[1], place_pos[2] + 0.02]
    ik_move(robot_id, ee_idx, place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # Release
    if constraint_id != -1:
        p.removeConstraint(constraint_id)

    # Retract
    ik_move(robot_id, ee_idx, above_place, p.getQuaternionFromEuler(approach_euler), arm_joint_indices, steps=120, client_id=client_id)

    # Record frames during the whole sequence
    for _ in range(360):
        p.stepSimulation()
        if _ % 2 == 0:
            recorder.add_frame(render_camera_frame(view, proj, w, h))

    recorder.close()
    p.disconnect(client_id)


if __name__ == "__main__":
    run()