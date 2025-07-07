import mujoco, mujoco.viewer, numpy as np

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        # apply a trivial PD controller
        data.ctrl[:] = -0.1*data.qvel - 0.5*data.qpos
        mujoco.mj_step(model, data)
        v.sync()                      # make viewer reflect the new state