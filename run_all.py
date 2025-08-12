import os
import json

from tasks.task1_reach import run as run1
from tasks.task2_path_tracing import run as run2
from tasks.task3_pick_place import run as run3
from tasks.task4_obstacle_pick_place import run as run4
from tasks.task5_stacking import run as run5

# Optional MuJoCo task
try:
    from mj.tasks_mj_humanoid import run as run_mj
    HAS_MJ = True
except Exception:
    HAS_MJ = False


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    metrics = []
    metrics.append(run1("outputs/task1_reach.mp4"))
    metrics.append(run2("outputs/task2_path_tracing.mp4"))
    metrics.append(run3("outputs/task3_pick_place.mp4"))
    metrics.append(run4("outputs/task4_obstacle_pick_place.mp4"))
    metrics.append(run5("outputs/task5_stacking.mp4"))

    if HAS_MJ:
        metrics.append(run_mj("outputs/mj_humanoid_stabilize.mp4"))

    # Aggregate results
    results_path = os.path.join("outputs", "results.json")
    with open(results_path, "w") as f:
        json.dump({"results": metrics, "all_success": all(m.get("success", False) for m in metrics)}, f, indent=2)

    print(f"Wrote aggregated metrics to {results_path}")