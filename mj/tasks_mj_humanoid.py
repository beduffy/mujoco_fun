import os
import time
from typing import Dict, Any, List

import numpy as np
import mujoco as mj
import imageio
from PIL import Image, ImageDraw


def _draw_frame(width: int, height: int, t: int, dev: float, dev_hist: List[float]) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 22, 26))
    draw = ImageDraw.Draw(img)

    # Axes for deviation plot
    margin = 40
    plot_w = width - 2 * margin
    plot_h = height // 3
    plot_x0 = margin
    plot_y0 = height - margin - plot_h
    draw.rectangle([plot_x0, plot_y0, plot_x0 + plot_w, plot_y0 + plot_h], outline=(100, 100, 120))

    # Normalize dev history to plot range
    if dev_hist:
        max_dev = max(1e-6, max(dev_hist))
        points = []
        for i, d in enumerate(dev_hist[-plot_w:]):
            x = plot_x0 + i
            y = plot_y0 + plot_h - int((d / max_dev) * (plot_h - 2)) - 1
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=(80, 180, 250), width=2)

    # Text overlays
    draw.text((margin, margin), f"t={t}", fill=(220, 220, 230))
    draw.text((margin, margin + 20), f"qpos deviation={dev:.4f}", fill=(220, 220, 230))

    return np.asarray(img, dtype=np.uint8)


def run(output_path: str = "outputs/mj_humanoid_stabilize.mp4") -> Dict[str, Any]:
    model = mj.MjModel.from_xml_path("humanoid.xml")
    data = mj.MjData(model)

    width, height = 720, 480

    # PD stabilization
    qpos_ref = data.qpos.copy()
    pos_error = np.empty(model.nv)

    writer = imageio.get_writer(output_path, fps=30)

    dev_hist: List[float] = []
    steps = 600
    kp, kd = 0.5, 0.1

    # Precompute mapping from actuators to dof indices
    act_dof_indices: List[int] = []
    for a in range(model.nu):
        # actuator_trnid gives (joint id, qpos id or -1); use joint id->dofadr
        j_id = model.actuator_trnid[a][0]
        dof_adr = model.jnt_dofadr[j_id]
        act_dof_indices.append(int(dof_adr))

    for t in range(steps):
        mj.mj_differentiatePos(model, pos_error, 1.0, qpos_ref, data.qpos)
        # Fill controls per actuator from corresponding dof
        for a in range(model.nu):
            di = act_dof_indices[a]
            data.ctrl[a] = kp * pos_error[di] - kd * data.qvel[di]
        mj.mj_step(model, data)
        dev = float(np.linalg.norm(data.qpos - qpos_ref))
        dev_hist.append(dev)
        frame = _draw_frame(width, height, t, dev, dev_hist)
        writer.append_data(frame)

    writer.close()

    dev = float(np.linalg.norm(data.qpos - qpos_ref))
    success = dev < 0.5
    return {
        "task": "mj_humanoid_stabilize",
        "success": bool(success),
        "qpos_deviation": dev,
        "threshold": 0.5,
        "output_video": output_path,
    }


if __name__ == "__main__":
    print(run())