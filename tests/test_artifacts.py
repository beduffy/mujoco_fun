import json
import os
from pathlib import Path

ART_DIR = Path("/workspace/outputs")


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def test_outputs_directory_exists():
    assert ART_DIR.exists(), "outputs directory missing"


def test_results_json_schema():
    results_path = ART_DIR / "results.json"
    assert results_path.exists(), "results.json missing (run run_all.py first)"
    data = read_json(results_path)
    assert isinstance(data.get("results"), list), "results must be a list"
    for item in data["results"]:
        assert "task" in item and isinstance(item["task"], str)
        assert "success" in item and isinstance(item["success"], bool)
        assert "output_video" in item and isinstance(item["output_video"], str)


def test_videos_exist():
    data = read_json(ART_DIR / "results.json")
    for item in data["results"]:
        vid = ART_DIR / Path(item["output_video"]).name
        assert vid.exists(), f"missing video: {vid}"


def test_pybullet_tasks_success_and_progress():
    data = read_json(ART_DIR / "results.json")
    res = {r["task"]: r for r in data["results"]}
    must_succeed = ["task1_reach", "task2_path_tracing", "task4_obstacle_pick_place", "task6_figure_eight", "task7_rrt_reach"]
    for t in must_succeed:
        assert res.get(t, {}).get("success", False), f"Expected success for {t}"
    # Progress checks for pick_place and stacking
    if "task3_pick_place" in res:
        assert res["task3_pick_place"]["cube_to_goal_distance"] < 0.35
    if "task5_stacking" in res:
        assert res["task5_stacking"]["cube_a_distance"] < 0.5
        assert res["task5_stacking"]["cube_b_distance"] < 3.0


def test_mujoco_tasks_present():
    data = read_json(ART_DIR / "results.json")
    if not data.get("has_mujoco", False):
        return
    mj = [r for r in data["results"] if r["task"].startswith("mj_")]
    names = {r["task"] for r in mj}
    assert "mj_reach2d" in names
    assert "mj_cartpole" in names