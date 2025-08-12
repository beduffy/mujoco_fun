import json
from pathlib import Path

from tools.collect_panda_reach import collect
from tools.train_bc_panda_reach import train_ls, evaluate, OUT_DIR


def test_bc_pipeline(tmp_path):
    # Collect small dataset
    collect(num_episodes=5, max_steps=50)
    # Train and evaluate
    W = train_ls()
    sr = evaluate(W, episodes=5, max_steps=50)
    # Save eval
    (OUT_DIR / "eval_test.json").write_text(json.dumps({"success_rate": sr}, indent=2))
    assert sr > 0.6, f"expected BC success_rate>0.6, got {sr}"