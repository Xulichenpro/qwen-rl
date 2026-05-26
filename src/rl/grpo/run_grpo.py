"""GRPO training for the Qwen math solver.

All hyper-parameters live in configs/train/grpo.yml. Run with:
    python -m src.rl.grpo.run_grpo --config configs/train/grpo.yml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.lora.config import resolve_torch_dtype
from src.rl.grpo.config import GrpoConfig, load_grpo_config
from src.rl.grpo.rewards import answer_reward, concise_reward, format_reward

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_model_dir(cfg: GrpoConfig) -> str:
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


def _load_prompt_dataset(path: Path, max_items: int = -1) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    if max_items >= 0:
        dataset = dataset.select(range(min(max_items, len(dataset))))
    required_columns = {"prompt", "gold_answer"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise KeyError(f"GRPO dataset missing columns: {sorted(missing)}")
    return dataset


def _load_policy_model(cfg: GrpoConfig):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = cfg.model.get("adapter_dir")
    if not adapter_dir:
        raise ValueError("model.adapter_dir is required so GRPO starts from SFT LoRA")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model["local_dir"],
        device_map=cfg.model["device_map"],
        torch_dtype=resolve_torch_dtype(cfg.model["torch_dtype"]),
    )
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)


def _build_grpo_args(cfg: GrpoConfig):
    from trl import GRPOConfig

    training = cfg.training
    return GRPOConfig(
        output_dir=training["output_dir"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        logging_steps=training["logging_steps"],
        num_train_epochs=training["num_train_epochs"],
        save_steps=training["save_steps"],
        eval_steps=training["eval_steps"],
        learning_rate=float(training["learning_rate"]),
        warmup_steps=training.get("warmup_steps", 0),
        lr_scheduler_type=training["lr_scheduler_type"],
        max_grad_norm=training["max_grad_norm"],
        bf16=training.get("bf16"),
        fp16=training.get("fp16", False),
        use_cpu=training.get("use_cpu", False),
        save_total_limit=training["save_total_limit"],
        save_on_each_node=training["save_on_each_node"],
        gradient_checkpointing=training["gradient_checkpointing"],
        report_to=training["report_to"],
        eval_strategy=training["eval_strategy"],
        save_strategy=training["save_strategy"],
        remove_unused_columns=False,
        do_train=True,
        do_eval=training["eval_strategy"] != "no",
        num_generations=training["num_generations"],
        max_completion_length=training["max_completion_length"],
        temperature=float(training["temperature"]),
        top_p=float(training["top_p"]),
        top_k=training.get("top_k", 0),
        beta=float(training["beta"]),
        epsilon=float(training["epsilon"]),
        reward_weights=training.get("reward_weights"),
        use_vllm=training.get("use_vllm", False),
        vllm_mode=training.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=training.get("vllm_gpu_memory_utilization", 0.3),
        log_completions=training.get("log_completions", False),
        num_completions_to_print=training.get("num_completions_to_print"),
    )


def _maybe_swanlab_callback(cfg: GrpoConfig):
    if not cfg.swanlab.get("enabled", False):
        return None
    from swanlab.integration.huggingface import SwanLabCallback

    return SwanLabCallback(
        project=cfg.swanlab["project"],
        experiment_name=cfg.swanlab["experiment_name"],
        config=cfg.swanlab["config"],
    )


def run(cfg: GrpoConfig) -> None:
    from modelscope import AutoTokenizer
    from trl import GRPOTrainer

    local_dir = _ensure_model_dir(cfg)
    cfg.model["local_dir"] = local_dir

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=cfg.model["use_fast_tokenizer"],
        trust_remote_code=cfg.model["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_policy_model(cfg)
    args = _build_grpo_args(cfg)
    train_dataset = _load_prompt_dataset(
        Path(cfg.data["train_path"]),
        max_items=int(cfg.data.get("train_max_items", -1)),
    )
    eval_dataset = _load_prompt_dataset(
        Path(cfg.data["eval_path"]),
        max_items=int(cfg.data.get("eval_max_items", -1)),
    )

    callbacks = []
    cb = _maybe_swanlab_callback(cfg)
    if cb is not None:
        callbacks.append(cb)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[answer_reward, format_reward, concise_reward],
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
        default=REPO_ROOT / "configs" / "train" / "grpo.yml",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="cap training and eval items (overrides data.*_max_items)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_grpo_config(args.config)
    if args.max_items is not None:
        cfg.data["train_max_items"] = args.max_items
        cfg.data["eval_max_items"] = args.max_items
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
