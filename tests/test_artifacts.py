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


def test_pybullet_tasks_success():
    data = read_json(ART_DIR / "results.json")
    pb = [r for r in data["results"] if r["task"].startswith("task")]  # task1..task5
    assert len(pb) == 5
    assert all(r["success"] for r in pb), f"PyBullet tasks should succeed: {pb}"


def test_mujoco_tasks_present():
    data = read_json(ART_DIR / "results.json")
    mj = [r for r in data["results"] if r["task"].startswith("mj_")]
    # At least the 2D reach and cart-pole should be present when mujoco installed
    names = {r["task"] for r in mj}
    assert "mj_reach2d" in names
    assert "mj_cartpole" in names