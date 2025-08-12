import numpy as np
import pybullet as p
from typing import List, Tuple


def move_ee_towards(
    robot_id: int,
    ee_idx: int,
    arm_joint_indices: List[int],
    target_pos: Tuple[float, float, float],
    client_id: int,
    kp: float = 3.0,
    damping: float = 1e-3,
    dt: float = 1.0 / 240.0,
    inner_steps: int = 4,
    iters: int = 60,
) -> None:
    # Build active (non-fixed) joint list once
    num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
    active_joint_indices: List[int] = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            active_joint_indices.append(j)
    # Map from joint index to position in active vector
    joint_to_active_pos = {j: i for i, j in enumerate(active_joint_indices)}
    # Nominal posture for nullspace (current posture)
    nominal_arm = np.array([p.getJointState(robot_id, j, physicsClientId=client_id)[0] for j in arm_joint_indices])

    for _ in range(iters):
        ee_state = p.getLinkState(robot_id, ee_idx, physicsClientId=client_id)
        ee_pos = np.array(ee_state[0])
        err = np.array(target_pos) - ee_pos
        v = kp * err
        v_norm = np.linalg.norm(v)
        if v_norm > 0.2:
            v = v * (0.2 / v_norm)

        # Current joint states for active DOF
        joint_states = p.getJointStates(robot_id, active_joint_indices, physicsClientId=client_id)
        q_active = np.array([s[0] for s in joint_states], dtype=np.float64)
        zeros = [0.0] * len(q_active)

        J_lin, J_ang = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=ee_idx,
            localPosition=[0, 0, 0],
            objPositions=q_active.tolist(),
            objVelocities=zeros,
            objAccelerations=zeros,
            physicsClientId=client_id,
        )
        J = np.array(J_lin)[:, : len(q_active)]  # 3 x dof_active

        # Select arm columns only
        arm_cols = [joint_to_active_pos[j] for j in arm_joint_indices]
        J_arm = J[:, arm_cols]  # 3 x dof_arm

        # Damped least squares for arm
        JJt = J_arm @ J_arm.T
        qdot_task = J_arm.T @ np.linalg.solve(JJt + damping * np.eye(3), v)
        # Nullspace towards nominal posture
        arm_states = p.getJointStates(robot_id, arm_joint_indices, physicsClientId=client_id)
        q_arm = np.array([s[0] for s in arm_states])
        null_dir = 0.1 * (nominal_arm - q_arm)
        N = np.eye(J_arm.shape[1]) - J_arm.T @ np.linalg.solve(JJt + damping * np.eye(3), J_arm)
        qdot_arm = qdot_task + N @ null_dir
        # Velocity limits
        vmax = 1.0
        qdot_arm = np.clip(qdot_arm, -vmax, vmax)

        q_arm_next = q_arm + qdot_arm * dt * inner_steps

        for i, j in enumerate(arm_joint_indices):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(q_arm_next[i]),
                force=300,
                positionGain=0.5,
                velocityGain=1.0,
                physicsClientId=client_id,
            )
        for _inner in range(inner_steps):
            p.stepSimulation(physicsClientId=client_id)