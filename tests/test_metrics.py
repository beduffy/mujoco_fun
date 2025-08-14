import json
from pathlib import Path

ART_DIR = Path("/workspace/outputs")


def test_mj_reach2d_success():
    data = json.loads((ART_DIR / "results.json").read_text())
    if not data.get("has_mujoco", False):
        return
    r = next(r for r in data["results"] if r["task"] == "mj_reach2d")
    assert r["success"], f"mj_reach2d should succeed: {r}"


def test_mj_cartpole_bounds():
    data = json.loads((ART_DIR / "results.json").read_text())
    if not data.get("has_mujoco", False):
        return
    r = next(r for r in data["results"] if r["task"] == "mj_cartpole")
    assert r["pos_abs"] < 1.5, "cart should stay within bounds"