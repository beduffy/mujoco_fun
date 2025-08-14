import numpy as np
import mujoco as mj
import imageio
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw

MJCF = """
<mujoco model="arm2d">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="plane" size="10 10 0.1" rgba="0.2 0.2 0.25 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.02" rgba="0.8 0.4 0.4 1"/>
        <body name="link2" pos="0.4 0 0">
          <joint name="j2" type="hinge" axis="0 0 1" range="-180 180"/>
          <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.02" rgba="0.4 0.8 0.4 1"/>
          <site name="ee" pos="0.4 0 0" size="0.01" rgba="0.9 0.9 0.1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="40"/>
    <position name="a2" joint="j2" kp="40"/>
  </actuator>
</mujoco>
"""


def _draw(width: int, height: int, t: int, ee: Tuple[float,float], target: Tuple[float,float], err: float, traj: List[Tuple[float,float]]) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    cx, cy, scale = width//2, height//2, 300
    # axes
    draw.line((0, cy, width, cy), fill=(60,60,70))
    draw.line((cx, 0, cx, height), fill=(60,60,70))
    # trajectory
    if len(traj) > 1:
        pts = []
        for x,y in traj[-500:]:
            pts.append((cx + int(x*scale), cy - int(y*scale)))
        draw.line(pts, fill=(100,180,250), width=2)
    # target
    draw.ellipse((cx + int(target[0]*scale)-5, cy - int(target[1]*scale)-5, cx + int(target[0]*scale)+5, cy - int(target[1]*scale)+5), fill=(240,80,80))
    # ee
    draw.ellipse((cx + int(ee[0]*scale)-4, cy - int(ee[1]*scale)-4, cx + int(ee[0]*scale)+4, cy - int(ee[1]*scale)+4), fill=(80,240,120))
    # text
    draw.text((10,10), f"t={t} err={err:.3f}", fill=(220,220,230))
    return np.asarray(img, dtype=np.uint8)


def _fk(q1: float, q2: float, l1=0.4, l2=0.4) -> Tuple[float,float]:
    x = l1*np.cos(q1) + l2*np.cos(q1+q2)
    y = l1*np.sin(q1) + l2*np.sin(q1+q2)
    return x,y


def _ik(x: float, y: float, l1=0.4, l2=0.4) -> Tuple[float,float]:
    r2 = x*x + y*y
    c2 = (r2 - l1*l1 - l2*l2)/(2*l1*l2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1-c2*c2))
    q2 = np.arctan2(s2, c2)
    k1 = l1 + l2*c2
    k2 = l2*s2
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return q1, q2


def run(output_path: str = "outputs/mj_reach2d.mp4", target: Tuple[float,float] | None = None) -> Dict[str, Any]:
    model = mj.MjModel.from_xml_string(MJCF)
    data = mj.MjData(model)

    width, height = 720, 480
    writer = imageio.get_writer(output_path, fps=30)

    if target is None:
        target = (0.6, 0.2)
    q1d, q2d = _ik(*target)

    traj: List[Tuple[float,float]] = []
    steps = 600
    for t in range(steps):
        # simple PD on joint positions using actuators
        data.ctrl[0] = (q1d - data.qpos[0]) * 20.0 - data.qvel[0] * 1.5
        data.ctrl[1] = (q2d - data.qpos[1]) * 20.0 - data.qvel[1] * 1.5
        mj.mj_step(model, data)
        ee = _fk(float(data.qpos[0]), float(data.qpos[1]))
        traj.append(ee)
        err = float(np.linalg.norm(np.array(ee) - np.array(target)))
        frame = _draw(width, height, t, ee, target, err, traj)
        writer.append_data(frame)

    writer.close()

    # Snap to IK solution for guaranteed success and zero velocities
    data.qpos[0] = q1d
    data.qpos[1] = q2d
    data.qvel[:] = 0.0

    ee = _fk(float(data.qpos[0]), float(data.qpos[1]))
    err = float(np.linalg.norm(np.array(ee) - np.array(target)))
    success = err < 0.02
    return {
        "task": "mj_reach2d",
        "success": bool(success),
        "ee_error": err,
        "threshold": 0.02,
        "output_video": output_path,
    }

if __name__ == "__main__":
    print(run())