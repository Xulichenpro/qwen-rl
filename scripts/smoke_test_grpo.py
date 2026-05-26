"""Smoke test for GRPO config + CLI imports.

No GPU required. Verifies:
1. configs/train/grpo.yml parses + jinja2-renders without errors.
2. Reward weights match the reward function list used by run_grpo.
3. `python -m src.rl.grpo.run_grpo --help` imports cleanly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rl.grpo.config import load_grpo_config  # noqa: E402

CFG_PATH = REPO_ROOT / "configs" / "train" / "grpo.yml"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_config_loads() -> None:
    cfg = load_grpo_config(CFG_PATH)
    for section in ("model", "data", "training", "swanlab"):
        _assert(hasattr(cfg, section), f"config missing section {section}")
    expected_local = f"{cfg.model['cache_dir']}{cfg.model['id']}/"
    _assert(
        cfg.model["local_dir"] == expected_local,
        f"local_dir not interpolated: {cfg.model['local_dir']!r}",
    )
    _assert(
        cfg.swanlab["config"]["base_adapter"] == cfg.model["adapter_dir"],
        "swanlab.config.base_adapter not interpolated",
    )
    print("ok  config_loads")


def test_reward_weights_match_reward_functions() -> None:
    cfg = load_grpo_config(CFG_PATH)
    _assert(
        len(cfg.training["reward_weights"]) == 3,
        "reward_weights should match answer/format/concise reward functions",
    )
    print("ok  reward_weights")


def test_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.rl.grpo.run_grpo", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    _assert(
        proc.returncode == 0,
        f"run_grpo --help exit {proc.returncode}\nstderr:\n{proc.stderr}",
    )
    _assert("--config" in proc.stdout, "run_grpo --help missing --config flag")
    _assert("--max-items" in proc.stdout, "run_grpo --help missing --max-items flag")
    print("ok  cli_help")


if __name__ == "__main__":
    test_config_loads()
    test_reward_weights_match_reward_functions()
    test_cli_help()
    print("\nALL GRPO SMOKE TESTS PASSED")
