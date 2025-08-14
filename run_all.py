import os
import json
import argparse

from tasks.task1_reach import run as run1
from tasks.task2_path_tracing import run as run2
from tasks.task3_pick_place import run as run3
from tasks.task4_obstacle_pick_place import run as run4
from tasks.task5_stacking import run as run5
from tasks.task6_figure_eight import run as run6
from tasks.task7_rrt_reach import run as run7
from tasks.task8_rrt_joint_reach import run as run8

# Optional MuJoCo tasks
HAS_MJ = False
try:
    from mj.tasks_mj_humanoid import run as run_mj_hum
    from mj.tasks_mj_reach2d import run as run_mj_arm
    from mj.tasks_mj_cartpole import run as run_mj_cart
    HAS_MJ = True
except Exception:
    HAS_MJ = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mujoco", action="store_true", help="Skip MuJoCo tasks")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    metrics = []
    metrics.append(run1("outputs/task1_reach.mp4"))
    metrics.append(run2("outputs/task2_path_tracing.mp4"))
    metrics.append(run3("outputs/task3_pick_place.mp4"))
    metrics.append(run4("outputs/task4_obstacle_pick_place.mp4"))
    metrics.append(run5("outputs/task5_stacking.mp4"))
    metrics.append(run6("outputs/task6_figure_eight.mp4"))
    metrics.append(run7("outputs/task7_rrt_reach.mp4"))
    metrics.append(run8("outputs/task8_rrt_joint_reach.mp4"))

    included_mj = False
    if HAS_MJ and not args.no_mujoco:
        metrics.append(run_mj_arm("outputs/mj_reach2d.mp4"))
        metrics.append(run_mj_cart("outputs/mj_cartpole.mp4"))
        metrics.append(run_mj_hum("outputs/mj_humanoid_stabilize.mp4"))
        included_mj = True

    results_path = os.path.join("outputs", "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "results": metrics,
            "all_success": all(m.get("success", False) for m in metrics),
            "has_mujoco": bool(included_mj)
        }, f, indent=2)

    print(f"Wrote aggregated metrics to {results_path}")