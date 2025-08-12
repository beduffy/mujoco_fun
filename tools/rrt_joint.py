import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Tuple


class JointSpacePlanner:
    def __init__(self, obstacle_size: Tuple[float, float, float], obstacle_pos: Tuple[float, float, float]):
        self.cid = p.connect(p.DIRECT)
        p.resetSimulation(self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.cid)
        # create obstacle
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in obstacle_size], physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in obstacle_size], rgbaColor=[0.2,0.8,0.2,1], physicsClientId=self.cid)
        self.obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=obstacle_pos, physicsClientId=self.cid)
        # arm joint indices
        self.arm_joints = []
        self.link_indices = []
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            if info[2] != p.JOINT_FIXED and info[1].decode("utf-8").startswith("panda_joint"):
                self.arm_joints.append(j)
                self.link_indices.append(j)
        # joint limits
        lows, highs = [], []
        for j in self.arm_joints:
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            lows.append(info[8])
            highs.append(info[9])
        self.lower = np.array(lows)
        self.upper = np.array(highs)

    def __del__(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass

    def set_q(self, q: np.ndarray):
        for i, j in enumerate(self.arm_joints):
            p.resetJointState(self.robot, j, float(q[i]), physicsClientId=self.cid)

    def collision(self) -> bool:
        # check minimal distance between links and obstacle
        for link in self.link_indices:
            pts = p.getClosestPoints(self.robot, self.obstacle, distance=0.005, linkIndexA=link, physicsClientId=self.cid)
            if pts:
                return True
        return False

    def rrt(self, q_start: np.ndarray, q_goal: np.ndarray, max_iters: int = 4000, step: float = 0.1) -> List[np.ndarray]:
        rng = np.random.default_rng(0)
        nodes = [q_start.copy()]
        parents = [-1]

        def nearest(qs: List[np.ndarray], q: np.ndarray) -> int:
            dists = [np.linalg.norm(qi - q) for qi in qs]
            return int(np.argmin(dists))

        def steer(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
            d = q_to - q_from
            n = np.linalg.norm(d)
            if n < 1e-9:
                return q_from.copy()
            return q_from + d / n * step

        def valid(q: np.ndarray) -> bool:
            q_clip = np.clip(q, self.lower, self.upper)
            self.set_q(q_clip)
            return not self.collision()

        if not valid(q_start) or not valid(q_goal):
            return []

        for it in range(max_iters):
            sample = q_goal if rng.random() < 0.2 else self.lower + rng.random(len(self.arm_joints)) * (self.upper - self.lower)
            idx = nearest(nodes, sample)
            q_new = steer(nodes[idx], sample)
            if valid(q_new):
                parents.append(idx)
                nodes.append(q_new)
                if np.linalg.norm(q_new - q_goal) < step and valid(q_goal):
                    # reconstruct
                    path = [q_goal.copy(), q_new.copy()]
                    cur = len(nodes) - 1
                    while parents[cur] != -1:
                        cur = parents[cur]
                        path.append(nodes[cur].copy())
                    return list(reversed(path))
        return []

    def shortcut(self, path: List[np.ndarray], trials: int = 200) -> List[np.ndarray]:
        if len(path) < 3:
            return path
        rng = np.random.default_rng(0)
        pts = [p.copy() for p in path]
        for _ in range(trials):
            if len(pts) < 3:
                break
            i = rng.integers(0, len(pts)-2)
            j = rng.integers(i+2, len(pts))
            # check segment validity by interpolating
            ok = True
            a, b = pts[i], pts[j]
            for alpha in np.linspace(0, 1, num=10):
                q = (1-alpha)*a + alpha*b
                if not self._valid_only(q):
                    ok = False
                    break
            if ok:
                pts = pts[:i+1] + pts[j:]
        return pts

    def _valid_only(self, q: np.ndarray) -> bool:
        q_clip = np.clip(q, self.lower, self.upper)
        self.set_q(q_clip)
        return not self.collision()