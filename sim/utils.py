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

    # Plane only for simplicity and robust manipulation
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    return client_id


def get_default_camera(width: int = 720, height: int = 480) -> Tuple[List[float], List[float], int, int]:
    # Camera looking towards the workspace around (0.5, 0, 0.2)
    distance = 1.2
    yaw = 45
    pitch = -35
    target_pos = [0.5, 0.0, 0.2]
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

    # Map panda_jointN -> index
    name_to_index = {}
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        joint_name = info[1].decode("utf-8")
        name_to_index[joint_name] = j
        if joint_name.startswith("panda_finger_joint"):
            finger_joint_indices.append(j)
        if joint_name in ("panda_hand", "panda_hand_tcp"):
            end_effector_link_index = j

    # Explicit arm joint order
    for k in range(1, 8):
        jname = f"panda_joint{k}"
        if jname in name_to_index:
            arm_joint_indices.append(name_to_index[jname])

    if end_effector_link_index is None:
        end_effector_link_index = name_to_index.get("panda_hand", min(num_joints - 1, 11))

    # Damping for stability
    for j in arm_joint_indices + finger_joint_indices:
        p.changeDynamics(robot_id, j, linearDamping=0.04, angularDamping=0.04, physicsClientId=client_id)

    return robot_id, arm_joint_indices, finger_joint_indices, end_effector_link_index


def open_gripper(robot_id: int, finger_joint_indices: List[int], width: float = 0.04, client_id: int = 0) -> None:
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
    # Fetch joint limits for arm joints
    lower_limits: List[float] = []
    upper_limits: List[float] = []
    joint_ranges: List[float] = []
    rest_poses: List[float] = []
    for j in arm_joint_indices:
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        joint_ranges.append(upper_limits[-1] - lower_limits[-1])
        rest_poses.append((lower_limits[-1] + upper_limits[-1]) / 2.0)

    kwargs = dict(lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses, maxNumIterations=200, residualThreshold=1e-4, physicsClientId=client_id)
    if target_orn is None:
        joint_positions = p.calculateInverseKinematics(robot_id, end_eff_idx, target_pos, **kwargs)
    else:
        joint_positions = p.calculateInverseKinematics(robot_id, end_eff_idx, target_pos, target_orn, **kwargs)

    for i, j in enumerate(arm_joint_indices):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=joint_positions[i], force=300, positionGain=0.3, velocityGain=1.0, physicsClientId=client_id)
        # Hard-set joint state to ensure convergence in headless mode
        p.resetJointState(robot_id, j, joint_positions[i], targetVelocity=0.0, physicsClientId=client_id)

    step_simulation(steps, client_id)


def wait_until_settled(body_id: int, client_id: int, lin_vel_thresh: float = 0.02, ang_vel_thresh: float = 0.05, stable_steps: int = 20, max_steps: int = 600) -> bool:
    stable = 0
    for i in range(max_steps):
        lin, ang = p.getBaseVelocity(body_id, physicsClientId=client_id)
        lin_n = 0.0 if lin is None else float(np.linalg.norm(lin))
        ang_n = 0.0 if ang is None else float(np.linalg.norm(ang))
        if lin_n < lin_vel_thresh and ang_n < ang_vel_thresh:
            stable += 1
        else:
            stable = 0
        p.stepSimulation(physicsClientId=client_id)
        if stable >= stable_steps:
            return True
    return False


def get_body_z(body_id: int, client_id: int) -> float:
    pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    return float(pos[2])