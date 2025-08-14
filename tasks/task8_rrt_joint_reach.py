import numpy as np
import pybullet as p
from typing import Dict, Any

from sim.utils import setup_simulation, get_default_camera, render_camera_frame, VideoRecorder, load_panda
from tools.rrt_joint import JointSpacePlanner


def run(output_path: str = "outputs/task8_rrt_joint_reach.mp4") -> Dict[str, Any]:
    cid = setup_simulation(gui=False)
    robot_id, arm_joint_indices, finger_joint_indices, ee_idx = load_panda(cid)
    view, proj, w, h = get_default_camera(720, 480)
    recorder = VideoRecorder(output_path, fps=30)

    obstacle_size = (0.12, 0.12, 0.20)
    obstacle_pos = (0.60, 0.00, 0.20)

    planner = JointSpacePlanner(obstacle_size, obstacle_pos)

    # Derive arm joints by name to avoid API differences
    arm_joints_task = []
    for j in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, j, physicsClientId=cid)
        if info[2] != p.JOINT_FIXED and info[1].decode("utf-8").startswith("panda_joint"):
            arm_joints_task.append(j)

    q_start = np.array([p.getJointState(robot_id, j, physicsClientId=cid)[0] for j in arm_joints_task])

    # Build a reachable goal
    goal_ee = [0.68, 0.12, 0.30]
    ee_state = p.getLinkState(robot_id, ee_idx, physicsClientId=cid)
    target_orn = ee_state[1]
    q_goal = None
    try:
        q_goal_full = p.calculateInverseKinematics(robot_id, ee_idx, goal_ee, targetOrientation=target_orn, physicsClientId=cid)
        q_goal = np.array([q_goal_full[i] for i in range(len(arm_joints_task))])
    except Exception:
        pass
    # Fallback to a ready pose if IK failed
    if q_goal is None or not np.isfinite(q_goal).all():
        q_goal = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])

    path = planner.rrt(q_start, q_goal, max_iters=2000, step=0.15)
    if path:
        path = planner.shortcut(path)

    if not path:
        # fallback: direct joint motion
        p.setJointMotorControlArray(robot_id, arm_joints_task, p.POSITION_CONTROL, targetPositions=q_goal.tolist(), physicsClientId=cid)
        for _ in range(240):
            p.stepSimulation(physicsClientId=cid)
            if _ % 3 == 0:
                recorder.add_frame(render_camera_frame(view, proj, w, h))
    else:
        for q in path:
            p.setJointMotorControlArray(robot_id, arm_joints_task, p.POSITION_CONTROL, targetPositions=q.tolist(), physicsClientId=cid)
            for _ in range(60):
                p.stepSimulation(physicsClientId=cid)
                if _ % 3 == 0:
                    recorder.add_frame(render_camera_frame(view, proj, w, h))

    final = np.array(p.getLinkState(robot_id, ee_idx, physicsClientId=cid)[0])
    dist = float(np.linalg.norm(final - np.array(goal_ee)))
    success = dist < 0.08

    recorder.close()
    p.disconnect(cid)
    return {
        "task": "task8_rrt_joint_reach",
        "success": bool(success),
        "ee_goal_distance": dist,
        "threshold": 0.08,
        "output_video": output_path,
    }


if __name__ == "__main__":
    print(run())