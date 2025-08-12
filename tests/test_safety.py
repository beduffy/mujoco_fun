import numpy as np
from sim.safety import clamp_joint_positions, clamp_joint_velocities, rate_limit
from sim.domain_randomization import randomize_masses, randomize_friction, randomize_camera
import pybullet as p
import pybullet_data

def test_safety_utils():
    q = np.array([0.0, 1.0, -2.0])
    lower = np.array([-0.5, 0.0, -1.0])
    upper = np.array([0.5, 0.5, 1.0])
    qc = clamp_joint_positions(q, lower, upper)
    assert qc.tolist() == [0.0, 0.5, -1.0]
    qdot = clamp_joint_velocities(np.array([2.0, -3.0]), vmax=1.0)
    assert np.all(np.abs(qdot) <= 1.0)
    a = rate_limit(np.array([0.0, 0.0]), np.array([1.0, 0.0]), max_step=0.25)
    assert np.allclose(a, np.array([0.25, 0.0]))


def test_domain_randomization_smoke():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    b = p.loadURDF("plane.urdf", physicsClientId=cid)
    randomize_masses([b], client_id=cid)
    randomize_friction([b], client_id=cid)
    dx, dy, dz = randomize_camera()
    assert -0.1 <= dx <= 0.1
    p.disconnect(cid)