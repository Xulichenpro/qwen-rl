"""LoRA-adapter inference for Qwen2.5-0.5B-Instruct on test.json.

All parameters live in configs/train/lora.yml. Run with:
    python -m src.lora.infer --config configs/train/lora.yml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import LoraConfig, load_lora_config, resolve_torch_dtype
from .qwen_ft import INSTRUCTION

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_messages(row: dict) -> list[dict[str, str]]:
    """Build the inference prompt using the same system instruction as SFT."""
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": row["question"]},
    ]


def predict(messages, model, tokenizer, *, device: str, max_new_tokens: int, do_sample: bool) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run(cfg: LoraConfig) -> None:
    from modelscope import AutoTokenizer
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    with open(cfg.data["test_path"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_max_items = int(cfg.data.get("test_max_items", -1))
    if test_max_items >= 0:
        test_data = test_data[:test_max_items]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model["local_dir"],
        use_fast=cfg.model["use_fast_tokenizer"],
        trust_remote_code=cfg.model["trust_remote_code"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model["local_dir"],
        device_map=cfg.model["device_map"],
        torch_dtype=resolve_torch_dtype(cfg.model["torch_dtype"]),
    )
    model = PeftModel.from_pretrained(model, model_id=cfg.inference["adapter_dir"])

    out_path = Path(cfg.inference["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = cfg.inference["device"]
    max_new_tokens = int(cfg.inference["max_new_tokens"])
    do_sample = bool(cfg.inference["do_sample"])

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(test_data, ncols=100, desc="infer"):
            messages = build_messages(row)
            response = predict(
                messages,
                model,
                tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            response = response.replace("\n", " ")
            f.write(f"{row['id']},{response}\n")


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
        help="cap inference items (overrides data.test_max_items)",
    )
    p.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="LoRA adapter directory to load (overrides inference.adapter_dir)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_lora_config(args.config)
    if args.max_items is not None:
        cfg.data["test_max_items"] = args.max_items
    if args.adapter_dir is not None:
        cfg.inference["adapter_dir"] = args.adapter_dir
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
