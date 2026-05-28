"""Readable DPO training entry point for the math solver.

Run from the repository root:

    python -m src.rl.dpo.train --config configs/train/dpo.yml

This script expects ``datasets/dpo_train/train.jsonl`` and ``val.jsonl`` to
already exist in TRL's preference format: prompt/chosen/rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.rl.dpo.config import MathDpoConfig, load_config
from src.rl.dpo.data import load_preference_dataset, preview_preference_file
from src.lora.config import resolve_torch_dtype


REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_model_dir(cfg: MathDpoConfig) -> str:
    """Return local base-model path, downloading from ModelScope if needed."""
    from modelscope import snapshot_download

    local_dir = Path(cfg.model["local_dir"])
    if local_dir.exists():
        return str(local_dir)

    snapshot_download(
        cfg.model["id"],
        cache_dir=cfg.model["cache_dir"],
        revision=cfg.model["revision"],
    )
    return cfg.model["local_dir"]


def load_tokenizer(local_dir: str, cfg: MathDpoConfig):
    from modelscope import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=cfg.model["use_fast_tokenizer"],
        trust_remote_code=cfg.model["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_sft_policy_model(local_dir: str, cfg: MathDpoConfig):
    """Load base Qwen and attach the SFT LoRA adapter as trainable weights."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = cfg.model.get("adapter_dir")
    if not adapter_dir:
        raise ValueError("model.adapter_dir is required for DPO after SFT")

    # 加载原始千问
    base_model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=cfg.model["device_map"],
        torch_dtype=resolve_torch_dtype(cfg.model["torch_dtype"]),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )
    # 挂 SFT adapter
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)


def build_training_args(cfg: MathDpoConfig):
    """Translate YAML training fields to TRL's DPOConfig."""
    from trl import DPOConfig

    t = cfg.training
    return DPOConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        logging_steps=t["logging_steps"],
        num_train_epochs=t["num_train_epochs"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        learning_rate=float(t["learning_rate"]),
        beta=float(t["beta"]),          # DPO优化强度
        max_length=t["max_length"],
        warmup_steps=t.get("warmup_steps", 0),
        lr_scheduler_type=t["lr_scheduler_type"],
        max_grad_norm=t["max_grad_norm"],
        bf16=t.get("bf16", False),
        fp16=t.get("fp16", False),
        use_cpu=t.get("use_cpu", False),
        save_total_limit=t["save_total_limit"],
        save_on_each_node=t["save_on_each_node"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to=t["report_to"],
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        remove_unused_columns=False,
        do_train=True,
        do_eval=t["eval_strategy"] != "no",
    )


def maybe_swanlab_callback(cfg: MathDpoConfig):
    if not cfg.swanlab.get("enabled", False):
        return None
    from swanlab.integration.huggingface import SwanLabCallback

    return SwanLabCallback(
        project=cfg.swanlab["project"],
        experiment_name=cfg.swanlab["experiment_name"],
        config=cfg.swanlab["config"],
    )


def print_dry_run(cfg: MathDpoConfig) -> None:
    """Show what would be trained without loading Qwen or starting DPO."""
    train_preview = preview_preference_file(cfg.data["train_path"])
    eval_preview = preview_preference_file(cfg.data["eval_path"])
    summary: dict[str, Any] = {
        "base_model": cfg.model["id"],
        "base_local_dir": cfg.model["local_dir"],
        "sft_adapter_dir": cfg.model.get("adapter_dir"),
        "train_path": str(train_preview.path),
        "train_total": train_preview.total,
        "eval_path": str(eval_preview.path),
        "eval_total": eval_preview.total,
        "output_dir": cfg.training["output_dir"],
        "beta": cfg.training["beta"],
        "learning_rate": cfg.training["learning_rate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[first train row]")
    print(f"prompt:   {train_preview.first_prompt}")
    print(f"chosen:   {train_preview.first_chosen}")
    print(f"rejected: {train_preview.first_rejected}")


def run(cfg: MathDpoConfig) -> None:
    from trl import DPOTrainer
    # 准备base model 路径
    local_dir = ensure_model_dir(cfg)
    cfg.model["local_dir"] = local_dir
    # 加载base model tokenizer
    tokenizer = load_tokenizer(local_dir, cfg)
    # 加载 base Qwen + SFT adapter
    model = load_sft_policy_model(local_dir, cfg)
    args = build_training_args(cfg)
    
    train_dataset = load_preference_dataset(
        cfg.data["train_path"],
        max_items=int(cfg.data.get("train_max_items", -1)),
    )
    eval_dataset = load_preference_dataset(
        cfg.data["eval_path"],
        max_items=int(cfg.data.get("eval_max_items", -1)),
    )

    callbacks = []
    swanlab_callback = maybe_swanlab_callback(cfg)
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=None,
    )
    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model(cfg.training["output_dir"])

    if cfg.swanlab.get("enabled", False):
        import swanlab

        swanlab.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train" / "dpo.yml",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="cap train/eval rows for a small run",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="override model.adapter_dir with a specific SFT checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="override training.output_dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/data and print a preview; do not load models",
    )
    parser.add_argument(
        "--no-swanlab",
        action="store_true",
        help="disable SwanLab callback for this run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_items is not None:
        cfg.data["train_max_items"] = args.max_items
        cfg.data["eval_max_items"] = args.max_items
    if args.adapter_dir is not None:
        cfg.model["adapter_dir"] = args.adapter_dir
    if args.output_dir is not None:
        cfg.training["output_dir"] = args.output_dir
    if args.no_swanlab:
        cfg.swanlab["enabled"] = False

    if args.dry_run:
        print_dry_run(cfg)
        return 0

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
