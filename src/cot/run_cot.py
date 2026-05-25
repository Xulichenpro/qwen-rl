"""Few-shot CoT inference against the Qwen2.5-0.5B-Instruct base model.

Usage:
    python -m src.cot.run_cot \
        --input  datasets/raw_train/test.json \
        --output outputs/submissions/submit_cot.csv \
        --prompt-yaml configs/prompt/cot.yml \
        --limit -1

The model is downloaded via modelscope (mirrors src/lora/qwen_ft.py) and
chat-templated using the tokenizer. Each test item is rendered with the
CoT prompt (few-shots inlined in the system message), generated, parsed,
and appended to a CSV of `<id>,<answer>` lines (same format as
src/lora/infer.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .parse import extract_answer
from .prompts import PromptBank

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SCENARIO = "cot"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("cot")


def load_model(model_dir: Path, dtype: str, device: str):
    """Lazy-import heavy deps so prompt-only smoke tests don't pay for them."""
    import torch
    from modelscope import AutoTokenizer
    from transformers import AutoModelForCausalLM

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map=device,
        torch_dtype=torch_dtype,
    )
    model.eval()
    return tokenizer, model


def ensure_model_dir(model_id: str, cache_dir: Path) -> Path:
    """Download via modelscope if absent; return on-disk path."""
    from modelscope import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(model_id, cache_dir=str(cache_dir), revision="master")
    return Path(local)


def generate(
    messages: list[dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen_only = [
        o[len(i):] for i, o in zip(inputs.input_ids, out_ids)
    ]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw_train" / "test.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "submissions" / "submit_cot.csv",
    )
    p.add_argument(
        "--prompt-yaml",
        type=Path,
        default=REPO_ROOT / "configs" / "prompt" / "cot.yml",
    )
    p.add_argument(
        "--model-cache",
        type=Path,
        default=REPO_ROOT,
        help="modelscope cache_dir (matches src/lora/qwen_ft.py default of repo root)",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=-1, help="-1 = all")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="render prompts and exit; do not load the model.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()

    bank = PromptBank(args.prompt_yaml)
    log.info("prompt: %s", args.prompt_yaml)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]
    log.info("input=%s items=%d", args.input, len(data))

    if args.dry_run:
        sample = data[0]
        log.info("dry-run rendering for id=%s", sample.get("id"))
        pp = bank.render(SCENARIO, {"question": sample["question"]})
        print(f"[model_key]\n{pp.model_key}\n")
        print(f"[system]\n{pp.system}\n")
        print(f"[user]\n{pp.user}")
        return 0

    model_dir = ensure_model_dir(args.model_id, args.model_cache)
    log.info("model dir: %s", model_dir)
    tokenizer, model = load_model(model_dir, args.dtype, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for row in tqdm(data, ncols=100, desc="cot"):
            messages = bank.build_messages(
                SCENARIO, {"question": row["question"]}
            )
            raw = generate(messages, tokenizer, model, args.max_new_tokens)
            answer = extract_answer(raw)
            answer = answer.replace("\n", " ")
            out.write(f"{row['id']},{answer}\n")
    log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
