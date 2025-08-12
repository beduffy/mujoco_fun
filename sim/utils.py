import os
import math
import time
from typing import Tuple, List, Optional

import numpy as np
import pybullet as p
import pybullet_data
import imageio


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 30):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.writer = imageio.get_writer(output_path, fps=fps)
        self.fps = fps

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        # Expect HxWx3 uint8
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        self.writer.append_data(rgb_frame)

    def close(self) -> None:
        self.writer.close()


def setup_simulation(gui: bool = False, time_step: float = 1.0 / 240.0) -> int:
    connection_mode = p.GUI if gui else p.DIRECT
    client_id = p.connect(connection_mode)
    p.resetSimulation(physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setTimeStep(time_step, physicsClientId=client_id)

    # Plane and a simple table
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.625], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=client_id)

    return client_id


def get_default_camera(width: int = 720, height: int = 480) -> Tuple[List[float], List[float], int, int]:
    # Camera looking towards the table at (0.5, 0, 0)
    distance = 1.2
    yaw = 45
    pitch = -35
    target_pos = [0.5, 0.0, 0.0]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=target_pos, distance=distance, yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / float(height), nearVal=0.01, farVal=3.0)
    return view_matrix, proj_matrix, width, height


def render_camera_frame(view_matrix, proj_matrix, width: int, height: int) -> np.ndarray:
    img_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def load_panda(client_id: int, base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1)) -> Tuple[int, List[int], List[int], int]:
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=base_pos, baseOrientation=base_orn, useFixedBase=True, physicsClientId=client_id)

    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    arm_joint_indices: List[int] = []
    finger_joint_indices: List[int] = []
    end_effector_link_index: Optional[int] = None

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            arm_joint_indices.append(j)
        if "finger_joint" in joint_name:
            finger_joint_indices.append(j)
        if joint_name in ("panda_hand", "panda_hand_tcp"):
            end_effector_link_index = j

    if end_effector_link_index is None:
        # Fallback to the last link index
        end_effector_link_index = num_joints - 1

    # Set default damping/friction for stability
    for j in arm_joint_indices + finger_joint_indices:
        p.changeDynamics(robot_id, j, linearDamping=0.04, angularDamping=0.04, physicsClientId=client_id)

    return robot_id, arm_joint_indices[:7], finger_joint_indices, end_effector_link_index


def open_gripper(robot_id: int, finger_joint_indices: List[int], width: float = 0.04, client_id: int = 0) -> None:
    # Panda finger opening is symmetric; set target positions
    for j in finger_joint_indices:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=width, force=20, physicsClientId=client_id)


def close_gripper(robot_id: int, finger_joint_indices: List[int], client_id: int = 0) -> None:
    for j in finger_joint_indices:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=50, physicsClientId=client_id)


def step_simulation(steps: int, client_id: int) -> None:
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client_id)


def create_box(size: Tuple[float, float, float], mass: float, pos: Tuple[float, float, float], rgba=(1, 0, 0, 1), client_id: int = 0) -> int:
    collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=client_id)
    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=rgba, physicsClientId=client_id)
    body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=pos, physicsClientId=client_id)
    return body


def ik_move(robot_id: int, end_eff_idx: int, target_pos: Tuple[float, float, float], target_orn: Optional[Tuple[float, float, float, float]], arm_joint_indices: List[int], steps: int, client_id: int, null_space: bool = True) -> None:
    if target_orn is None:
        joint_positions = p.calculateInverseKinematics(robot_id, end_eff_idx, target_pos, physicsClientId=client_id)
    else:
        joint_positions = p.calculateInverseKinematics(robot_id, end_eff_idx, target_pos, target_orn, physicsClientId=client_id)

    # Apply to arm joints only (use first N joint positions)
    for i, j in enumerate(arm_joint_indices):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=joint_positions[i], force=200, physicsClientId=client_id)

    step_simulation(steps, client_id)