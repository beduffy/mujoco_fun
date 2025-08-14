import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, List

from PIL import Image, ImageDraw

from tasks.task1_reach import run as t1
from tasks.task2_path_tracing import run as t2
from tasks.task3_pick_place import run as t3
from tasks.task4_obstacle_pick_place import run as t4
from tasks.task5_stacking import run as t5
from tasks.task6_figure_eight import run as t6
from tasks.task7_rrt_reach import run as t7

OUT_DIR = Path("/workspace/outputs/benchmarks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "task1_reach": t1,
    "task2_path_tracing": t2,
    "task3_pick_place": t3,
    "task4_obstacle_pick_place": t4,
    "task5_stacking": t5,
    "task6_figure_eight": t6,
    "task7_rrt_reach": t7,
}


def draw_bar_chart(results: Dict[str, Any], out_path: Path) -> None:
    names = list(results.keys())
    vals = [results[n]["success_rate"] for n in names]
    width, height = 800, 400
    img = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(img)
    margin = 50
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    x0, y0 = margin, margin
    # axis
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(180, 180, 180))
    max_val = max(1e-6, max(vals))
    n = len(names)
    bar_w = plot_w // max(1, n)
    for i, (name, v) in enumerate(zip(names, vals)):
        x = x0 + i * bar_w + 5
        h = int((v / max_val) * (plot_h - 10))
        draw.rectangle([x, y0 + plot_h - h, x + bar_w - 10, y0 + plot_h], fill=(100, 180, 250))
        draw.text((x, y0 + plot_h + 5), name, fill=(20, 20, 20))
    img.save(out_path)


def benchmark(tasks: List[str], seeds: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    # JSONL for per-run
    jsonl_path = OUT_DIR / "runs.jsonl"
    with jsonl_path.open("w") as jf:
        for name in tasks:
            fn = TASKS[name]
            success_count = 0
            runtimes: List[float] = []
            for s in range(seeds):
                start = time.time()
                out = fn(f"outputs/{name}_bench_s{s}.mp4")
                dt = time.time() - start
                runtimes.append(dt)
                success_count += int(out.get("success", False))
                jf.write(json.dumps({"task": name, "seed": s, "success": bool(out.get("success", False)), "runtime_s": dt}) + "\n")
            rate = success_count / seeds
            summary[name] = {
                "success_rate": rate,
                "avg_runtime_s": sum(runtimes) / len(runtimes),
            }
    # write JSON
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    # write CSV
    with (OUT_DIR / "summary.csv").open("w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["task", "success_rate", "avg_runtime_s"])
        for k, v in summary.items():
            w.writerow([k, v["success_rate"], v["avg_runtime_s"]])
    # draw bar chart
    draw_bar_chart(summary, OUT_DIR / "summary.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS.keys()))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    summary = benchmark(args.tasks, args.seeds)
    print("Benchmark summary:", summary)