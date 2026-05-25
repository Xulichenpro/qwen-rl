"""Smoke test for the LoRA YAML config + loader.

No GPU required. Verifies:
1. configs/train/lora.yml parses + jinja2-renders without errors.
2. Derived paths (model.local_dir, swanlab.experiment_name, inference.adapter_dir)
   are interpolated from referenced fields.
3. dtype and TaskType strings map to real torch/peft enums.
4. `python -m src.lora.qwen_ft --help` and `... infer --help` import cleanly.
5. `build_process_func` truncates to data.max_length on a synthetic example.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lora.config import (  # noqa: E402
    load_lora_config,
    resolve_task_type,
    resolve_torch_dtype,
)
from src.lora.qwen_ft import build_process_func  # noqa: E402

CFG_PATH = REPO_ROOT / "configs" / "train" / "lora.yml"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_config_loads() -> None:
    cfg = load_lora_config(CFG_PATH)
    for s in ("model", "data", "lora", "training", "swanlab", "inference"):
        _assert(hasattr(cfg, s), f"config missing section {s}")
    _assert(cfg.model["id"] == "Qwen/Qwen2.5-0.5B-Instruct", "model.id mismatch")
    print("ok  config_loads")


def test_jinja_interpolation() -> None:
    cfg = load_lora_config(CFG_PATH)
    # local_dir should expand `{{ model.cache_dir }}{{ model.id }}/`.
    expected_local = f"{cfg.model['cache_dir']}{cfg.model['id']}/"
    _assert(
        cfg.model["local_dir"] == expected_local,
        f"local_dir not interpolated: {cfg.model['local_dir']!r} vs {expected_local!r}",
    )
    _assert(
        cfg.swanlab["experiment_name"] == cfg.model["id"],
        f"swanlab.experiment_name not interpolated: {cfg.swanlab['experiment_name']!r}",
    )
    _assert(
        cfg.swanlab["config"]["model"] == cfg.model["id"],
        "swanlab.config.model not interpolated",
    )
    expected_adapter = f"{cfg.training['output_dir']}/checkpoint-3750/"
    _assert(
        cfg.inference["adapter_dir"] == expected_adapter,
        f"adapter_dir not interpolated: {cfg.inference['adapter_dir']!r}",
    )
    print("ok  jinja_interpolation")


def test_resolvers() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("skip resolvers: torch not installed")
        return

    _assert(resolve_torch_dtype("bfloat16") is torch.bfloat16, "bfloat16 map")
    _assert(resolve_torch_dtype("fp16") is torch.float16, "fp16 map")
    try:
        resolve_torch_dtype("int4")
    except ValueError:
        pass
    else:
        _assert(False, "expected ValueError on unknown dtype")

    try:
        from peft import TaskType
    except ModuleNotFoundError:
        print("ok  resolvers (peft skipped)")
        return
    _assert(resolve_task_type("CAUSAL_LM") is TaskType.CAUSAL_LM, "task_type map")
    print("ok  resolvers")


def test_cli_help() -> None:
    for mod in ("src.lora.qwen_ft", "src.lora.infer"):
        proc = subprocess.run(
            [sys.executable, "-m", mod, "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        _assert(
            proc.returncode == 0,
            f"{mod} --help exit {proc.returncode}\nstderr:\n{proc.stderr}",
        )
        _assert("--config" in proc.stdout, f"{mod} --help missing --config flag")
    print("ok  cli_help")


def test_limit_fields_present() -> None:
    cfg = load_lora_config(CFG_PATH)
    _assert(
        "train_max_items" in cfg.data,
        "data.train_max_items must be declared in lora.yml",
    )
    _assert(
        "test_max_items" in cfg.data,
        "data.test_max_items must be declared in lora.yml",
    )
    print("ok  limit_fields_present")


def test_load_dataset_honors_limit() -> None:
    """Truncation in _load_dataset should respect max_items."""
    from src.lora.qwen_ft import _load_dataset

    rows = [
        {"instruction": "ins", "question": f"q{i}", "answer": str(i)}
        for i in range(7)
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(rows, tf)
        tmp_path = Path(tf.name)

    try:
        # Identity process_func so we can count items easily.
        ds_all = _load_dataset(tmp_path, lambda r: r, max_items=-1)
        _assert(len(ds_all) == 7, f"max_items=-1 should keep all, got {len(ds_all)}")
        ds_three = _load_dataset(tmp_path, lambda r: r, max_items=3)
        _assert(len(ds_three) == 3, f"max_items=3 should yield 3, got {len(ds_three)}")
        ds_zero = _load_dataset(tmp_path, lambda r: r, max_items=0)
        _assert(len(ds_zero) == 0, f"max_items=0 should yield 0, got {len(ds_zero)}")
    finally:
        tmp_path.unlink()
    print("ok  load_dataset_honors_limit")


def test_cli_max_items_override() -> None:
    """--max-items should override cfg.data.train_max_items."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.lora.qwen_ft", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    _assert(
        "--max-items" in proc.stdout,
        f"qwen_ft --help should expose --max-items\nstdout:\n{proc.stdout}",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "src.lora.infer", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    _assert(
        "--max-items" in proc.stdout,
        f"infer --help should expose --max-items\nstdout:\n{proc.stdout}",
    )
    print("ok  cli_max_items_override")


def test_process_func_truncation() -> None:
    cfg = load_lora_config(CFG_PATH)

    class FakeTok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            n = len(text)
            return {"input_ids": [1] * n, "attention_mask": [1] * n}

    pf = build_process_func(FakeTok(), cfg.data)
    out = pf({"instruction": "ins" * 200, "question": "q" * 200, "answer": "a" * 200})
    max_len = cfg.data["max_length"]
    _assert(
        len(out["input_ids"]) == max_len,
        f"input_ids not truncated to {max_len}, got {len(out['input_ids'])}",
    )
    _assert(
        len(out["labels"]) == max_len,
        f"labels not truncated to {max_len}, got {len(out['labels'])}",
    )
    print("ok  process_func_truncation")


if __name__ == "__main__":
    test_config_loads()
    test_jinja_interpolation()
    test_resolvers()
    test_cli_help()
    test_limit_fields_present()
    test_load_dataset_honors_limit()
    test_cli_max_items_override()
    test_process_func_truncation()
    print("\nALL SMOKE TESTS PASSED")
