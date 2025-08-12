# Sim2Real Checklist

1) Robot kinematics and URDF
- Load the real robot URDF; verify joint names/limits match sim
- Perform base and hand-eye calibration (AprilTags/ChArUco)
- Validate FK/IK equivalence on known poses

2) Safety
- Set joint velocity/accel/jerk limits
- Implement rate limiter and emergency stop handling
- Soft limits for workspace and collision zones

3) Dynamics and friction
- Run system ID for inertial parameters and joint friction
- Configure domain randomization ranges around identified values

4) Controllers
- Start with Cartesian impedance (K/B), add nullspace posture
- Clamp torques/velocities, handle singularities with adaptive damping
- Test at low gains before ramping up

5) Perception
- Swap synthetic render with real RGB/depth; apply camera calibration
- Add noise augmentation and lighting randomization

6) Planning
- Integrate MoveIt2/OMPL for joint-space planning
- Validate collision world from real sensors

7) Policies
- Collect real demos to fine-tune BC; use DAgger for iterative improvement
- For RL: start with TD3+BC or CQL for stability on real data

8) Verification
- Dry-run with simulated hardware-in-the-loop at matched rates
- Logging: record commands, states, timestamps; replay in sim for regression tests

9) Deployment
- Containerized stack (Docker) with ROS2 nodes and policies
- Launch scripts to bring up drivers, controllers, and policy runner