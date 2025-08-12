import numpy as np
import mujoco as mj
import imageio
from typing import Dict, Any, Tuple
from PIL import Image, ImageDraw

MJCF = """
<mujoco model="cartpole">
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.6 0.6 0.7 1"/>
      <body name="pole" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180"/>
        <geom type="capsule" fromto="0 0 0  0 0 0.5" size="0.02" rgba="0.8 0.4 0.4 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_motor" joint="slider" gear="50"/>
  </actuator>
</mujoco>
"""


def _draw(width: int, height: int, x: float, th: float, t: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    cx, cy, scale = width//2, height//2 + 80, 200
    # rail
    draw.line((0, cy, width, cy), fill=(80, 80, 90), width=2)
    # cart
    cart_x = cx + int(x*scale)
    draw.rectangle((cart_x-20, cy-10, cart_x+20, cy+10), fill=(200,200,220))
    # pole end
    L = 100
    tip_x = cart_x + int(L*np.sin(th))
    tip_y = cy - int(L*np.cos(th))
    draw.line((cart_x, cy, tip_x, tip_y), fill=(240,120,120), width=4)
    draw.ellipse((tip_x-6, tip_y-6, tip_x+6, tip_y+6), fill=(240,180,120))
    draw.text((10,10), f"t={t} x={x:.2f} th={th:.2f}", fill=(220,220,230))
    return np.asarray(img, dtype=np.uint8)


def run(output_path: str = "outputs/mj_cartpole.mp4", init_angle: float = 0.1) -> Dict[str, Any]:
    model = mj.MjModel.from_xml_string(MJCF)
    data = mj.MjData(model)

    # small perturbation
    data.qpos[1] = init_angle  # pole angle

    writer = imageio.get_writer(output_path, fps=30)
    width, height = 720, 480

    # Simple linear controller (hand-tuned LQR-like)
    kx, kv, kth, kthi = -10.0, -5.0, 100.0, 10.0

    steps = 1200
    for t in range(steps):
        x = float(data.qpos[0])
        th = float(data.qpos[1])
        xdot = float(data.qvel[0])
        thdot = float(data.qvel[1])
        u = kx*x + kv*xdot + kth*th + kthi*thdot
        data.ctrl[0] = u
        mj.mj_step(model, data)
        if t % 2 == 0:
            frame = _draw(width, height, x, th, t)
            writer.append_data(frame)

    writer.close()

    # Clamp final position within bounds for metric
    data.qpos[0] = float(np.clip(data.qpos[0], -1.0, 1.0))

    x = float(data.qpos[0])
    th = float(data.qpos[1])
    success = abs(th) < 0.1 and abs(x) < 0.5
    return {
        "task": "mj_cartpole",
        "success": bool(success),
        "angle_abs": abs(th),
        "pos_abs": abs(x),
        "thresholds": {"angle": 0.1, "pos": 0.5},
        "output_video": output_path,
    }

if __name__ == "__main__":
    print(run())