import json
from pathlib import Path
from tools.perf_bench import main as perf_main

PERF_DIR = Path("/workspace/outputs/perf")


def test_perf_bench():
    perf = perf_main()
    # Thresholds
    assert perf["reach"]["final_error"] < 0.05
    assert perf["figure8"]["mean_error"] < 0.08
    assert perf["rrt"]["final_error"] < 0.08
    assert perf["rrt"]["min_clearance"] >= 0.0
    assert (PERF_DIR / "perf.json").exists()
    assert (PERF_DIR / "perf.png").exists()