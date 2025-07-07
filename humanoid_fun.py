import mujoco, mujoco.viewer, numpy as np
import time

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data  = mujoco.MjData(model)

# Stabilize at initial configuration.
qpos_ref = data.qpos.copy()
pos_error = np.empty(model.nv)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        step_start = time.time()

        # Compute position error using mj_differentiatePos.
        mujoco.mj_differentiatePos(model, pos_error, 1.0, qpos_ref, data.qpos)
        
        # apply a PD controller to actuated joints
        data.ctrl[:] = 0.5 * pos_error[6:] - 0.1 * data.qvel[6:]
        mujoco.mj_step(model, data)
        v.sync()                      # make viewer reflect the new state

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)