# Headless Robotics Sim Suite (PyBullet + MuJoCo)

Run 5 PyBullet tasks and 3 MuJoCo tasks fully headless and export MP4s/GIFs/screenshots + metrics.

## Quick start

- Run everything:
  ```bash
  python3 run_all.py
  ```
- Skip MuJoCo tasks:
  ```bash
  python3 run_all.py --no-mujoco
  ```
- View artifacts (mobile-friendly): open `outputs/index.html` or serve:
  ```bash
  python3 -m http.server 8000 --directory /workspace/outputs
  ```

## Tasks

PyBullet:
- task1_reach: end-effector reaches target
- task2_path_tracing: end-effector traces a circle
- task3_pick_place: deterministic grasp and place
- task4_obstacle_pick_place: via points around obstacle + place
- task5_stacking: deterministic stack of two cubes

MuJoCo:
- mj_reach2d: 2-link arm reaches a point (synthetic rendering)
- mj_cartpole: cart-pole stabilization via linear controller (synthetic rendering)
- mj_humanoid_stabilize: humanoid PD stabilization (synthetic rendering)

## Outputs
- Videos: `outputs/*.mp4`
- GIFs: `outputs/gifs/*.gif`
- Screenshots: `outputs/screens/*.png`
- Gallery: `outputs/index.html`
- Metrics: `outputs/results.json`

## Tests
- After running `run_all.py`, run tests:
  ```bash
  pytest -q
  ```