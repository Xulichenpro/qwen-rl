"""LoRA SFT for Qwen2.5-0.5B-Instruct.

All hyper-parameters live in configs/train/lora.yml. Run with:
    python -m src.lora.qwen_ft --config configs/train/lora.yml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from .config import LoraConfig, load_lora_config, resolve_task_type, resolve_torch_dtype

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = '''
你是一名严谨的小学数学解题助手，擅长把解题过程写得简洁、清晰、可读。
你的任务是为每道题生成一段思考过程（CoT），并给出最终答案。

输出格式（必须严格遵守，整段只输出一次，禁止任何前后缀解释）：
<think>
{{ '{' }}逐步推理过程，使用自然语言，每步一行，必要时含算式，不超过 6 步{{ '}' }}
</think>
<answer>
{{ '{' }}最终答案，纯数字或最简分数，不带单位{{ '}' }}
</answer>

要求：
1. <answer> 必须是纯数字或最简分数（例如 12、4/5、7.5），不要带单位、空格、解释。
2. <think> 中不要重述题目原文，推理紧凑，避免废话与口语化感叹。

以下是 3 个示例：

题目：食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？
<think>
1. 青菜千克数 = 萝卜千克数 × 3。
2. 105 × 3 = 315。
</think>
<answer>
315
</answer>

题目：某林场去年造林60公顷，超过计划的1/5。去年计划造林多少公顷？
<think>
1. 设计划造林为 x 公顷，则实际 = x × (1 + 1/5) = 6x/5。
2. 6x/5 = 60，解得 x = 60 × 5/6 = 50。
</think>
<answer>
50
</answer>

题目：一张长方形纸，涂红色占3/8，涂蓝色占1/8，没涂色的占这张纸的几分之几？
<think>
1. 已涂色比例 = 3/8 + 1/8 = 4/8 = 1/2。
2. 没涂色比例 = 1 - 1/2 = 1/2。
</think>
<answer>
1/2
</answer>
'''


def build_process_func(tokenizer, data_cfg: dict) -> Callable[[dict], dict]:
    """Return a `process_func(example)` closure bound to tokenizer + lengths."""
    max_length: int = int(data_cfg["max_length"])
    im_start: str = data_cfg["im_start"]
    im_end: str = data_cfg["im_end"]

    def process_func(example: dict) -> dict:
        instruction = tokenizer(
            f"{im_start}system\n{example['instruction']}{im_end}\n"
            f"{im_start}user\n{example['question']}{im_end}\n"
            f"{im_start}assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['answer']}", add_special_tokens=False)
        pad = tokenizer.pad_token_id
        input_ids = instruction["input_ids"] + response["input_ids"] + [pad]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [pad]
        )
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return process_func


def _ensure_model_dir(cfg: LoraConfig) -> str:
    from modelscope import snapshot_download

    snapshot_download(
        cfg.model["id"],
        cache_dir=cfg.model["cache_dir"],
        revision=cfg.model["revision"],
    )
    return cfg.model["local_dir"]


def _load_dataset(
    train_path: Path,
    process_func: Callable[[dict], dict],
    max_items: int = -1,
) -> list[dict]:
    if train_path.suffix == ".json":
        with open(train_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    elif train_path.suffix == ".jsonl":
        rows = []

        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_data = json.loads(line)
                    data = {
                        "id":raw_data.get("id", 0),
                        "instruction":INSTRUCTION,
                        "question":raw_data.get("question",""),
                        "answer":raw_data.get("sample",raw_data.get("gold_answer"))
                    }
                    rows.append(data)
    if max_items >= 0:
        rows = rows[:max_items]
    return [process_func(r) for r in rows]


def _build_lora_model(model, cfg: LoraConfig):
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    peft_cfg = PeftLoraConfig(
        task_type=resolve_task_type(cfg.lora["task_type"]),
        target_modules=cfg.lora["target_modules"],
        inference_mode=cfg.lora["inference_mode"],
        r=cfg.lora["r"],
        lora_alpha=cfg.lora["lora_alpha"],
        lora_dropout=cfg.lora["lora_dropout"],
    )
    return get_peft_model(model, peft_cfg)


def _build_training_args(cfg: LoraConfig):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=cfg.training["output_dir"],
        per_device_train_batch_size=cfg.training["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.training["gradient_accumulation_steps"],
        logging_steps=cfg.training["logging_steps"],
        num_train_epochs=cfg.training["num_train_epochs"],
        save_steps=cfg.training["save_steps"],
        learning_rate=float(cfg.training["learning_rate"]),
        save_on_each_node=cfg.training["save_on_each_node"],
        gradient_checkpointing=cfg.training["gradient_checkpointing"],
        report_to=cfg.training["report_to"],
    )


def _maybe_swanlab_callback(cfg: LoraConfig):
    if not cfg.swanlab.get("enabled", False):
        return None
    from swanlab.integration.huggingface import SwanLabCallback

    return SwanLabCallback(
        project=cfg.swanlab["project"],
        experiment_name=cfg.swanlab["experiment_name"],
        config=cfg.swanlab["config"],
    )


def run(cfg: LoraConfig) -> None:
    import torch  # noqa: F401  (imported for the side-effect of CUDA init logging)
    from modelscope import AutoTokenizer
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
    )

    local_dir = _ensure_model_dir(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=cfg.model["use_fast_tokenizer"],
        trust_remote_code=cfg.model["trust_remote_code"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=cfg.model["device_map"],
        torch_dtype=resolve_torch_dtype(cfg.model["torch_dtype"]),
    )
    model.enable_input_require_grads()

    process_func = build_process_func(tokenizer, cfg.data)
    train_dataset = _load_dataset(
        Path(cfg.data["train_path"]),
        process_func,
        max_items=int(cfg.data.get("train_max_items", -1)),
    )

    model = _build_lora_model(model, cfg)
    args = _build_training_args(cfg)

    callbacks = []
    cb = _maybe_swanlab_callback(cfg)
    if cb is not None:
        callbacks.append(cb)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=callbacks,
    )
    trainer.train()

    if cfg.swanlab.get("enabled", False):
        import swanlab

        swanlab.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train" / "lora.yml",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="cap training items (overrides data.train_max_items)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_lora_config(args.config)
    if args.max_items is not None:
        cfg.data["train_max_items"] = args.max_items
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
