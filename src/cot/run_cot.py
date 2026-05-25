"""Few-shot CoT inference + GLM judging on the raw training set.

We run Qwen2.5-0.5B over `datasets/raw_train/train.json` (which has gold
answers) so the judge model can flag Qwen failures; those become the
rejected ("bad") side of future DPO training.

Per-item flow:
  1. Render cot.yml with the question and chat-template it for Qwen.
  2. Generate Qwen's raw CoT and parse out its final answer.
  3. Render cot_judge.yml with (question, gold_answer, qwen_answer) and
     call the judge model (glm-5.1-w4a8 by default).
  4. Append one JSON line per item to a rotating batch log under
     outputs/logs/cot/run_<timestamp>/batch_<N>.log (100 entries per file).
  5. If judge says correct=0 (and JSON parsed cleanly), append a record
     {id, question, gold_answer, qwen_answer, qwen_raw} to
     datasets/dpo_train/bad_out.jsonl.

Usage:
    python -m src.cot.run_cot \
        --max-items 200 \
        --batch-size 100 \
        --device cpu --dtype fp32

Flags:
    --max-items N    cap items processed (-1 = all)
    --no-judge       skip the judge call (no bad_out append)
    --dry-run        render prompts and exit; do not load Qwen or call the judge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .logger import BatchJsonlLogger
from .parse import extract_answer, parse_judge_json
from .prompts import PromptBank

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
COT_SCENARIO = "cot"
JUDGE_SCENARIO = "judge"


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


def make_judge_client(judge_bank: PromptBank, model_yaml: Path):
    """Construct a ChatClient for the judge scenario. Lazy-imports langchain."""
    from src.data_syn.clients import ChatClient, load_endpoints

    endpoints = load_endpoints(model_yaml)
    node = judge_bank.get_node(JUDGE_SCENARIO)
    return ChatClient(
        endpoints[node["model_key"]],
        temperature=float(node.get("temperature", 0.0)),
        max_tokens=int(node.get("max_tokens", 1024)),
    )


def judge_one(
    judge_bank: PromptBank,
    judge_client,
    question: str,
    gold_answer: str,
    qwen_answer: str,
) -> tuple[dict[str, Any], str, str]:
    """Render judge prompt, call the judge, parse JSON.

    Returns (parsed, raw, error). `error` is '' on success, else a short
    string. `raw` is the merged content+reasoning for traceability.
    """
    try:
        jpp = judge_bank.render(
            JUDGE_SCENARIO,
            {
                "question": question,
                "gold_answer": gold_answer,
                "qwen_answer": qwen_answer,
            },
        )
        reply = judge_client.chat(jpp.system, jpp.user)
        parsed = parse_judge_json(reply.content, reply.reasoning)
        return parsed, reply.merged(), ""
    except Exception as exc:  # noqa: BLE001
        return (
            {
                "parsed": False,
                "gold": gold_answer,
                "qwen": qwen_answer,
                "correct": 0,
                "reason": f"judge: {type(exc).__name__}",
            },
            "",
            f"{type(exc).__name__}: {exc}",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "raw_train" / "train.json",
        help="JSON file of items with id/question/answer fields",
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
        "--judge-yaml",
        type=Path,
        default=REPO_ROOT / "configs" / "prompt" / "cot_judge.yml",
    )
    p.add_argument(
        "--model-yaml",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "models.yml",
    )
    p.add_argument(
        "--bad-out",
        type=Path,
        default=REPO_ROOT / "datasets" / "dpo_train" / "bad_out.jsonl",
    )
    p.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "logs" / "cot",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="entries per batch_<N>.log file",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=-1,
        help="max number of test items to process (-1 = all)",
    )
    p.add_argument(
        "--model-cache",
        type=Path,
        default=REPO_ROOT,
        help="modelscope cache_dir (matches src/lora/qwen_ft.py default)",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="skip Kimi judging and bad_out append",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="render prompts (incl. judge) and exit; do not call Qwen or Kimi",
    )
    return p.parse_args()


def _apply_limit(data: list, max_items: int) -> list:
    if max_items < 0:
        return data
    return data[:max_items]


def main() -> int:
    args = parse_args()
    log = setup_logging()

    bank = PromptBank(args.prompt_yaml)
    judge_bank = None if args.no_judge else PromptBank(args.judge_yaml)
    log.info("prompt: %s | judge: %s", args.prompt_yaml, args.judge_yaml if judge_bank else "disabled")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    full = len(data)
    data = _apply_limit(data, args.max_items)
    log.info("input=%s items=%d (of %d, max_items=%d)", args.input, len(data), full, args.max_items)

    if args.dry_run:
        sample = data[0]
        log.info("dry-run rendering for id=%s", sample.get("id"))
        pp = bank.render(COT_SCENARIO, {"question": sample["question"]})
        print(f"[cot.system]\n{pp.system}\n")
        print(f"[cot.user]\n{pp.user}\n")
        if judge_bank is not None:
            jpp = judge_bank.render(
                JUDGE_SCENARIO,
                {
                    "question": sample["question"],
                    "gold_answer": str(sample.get("answer", "<unknown>")),
                    "qwen_answer": "<placeholder>",
                },
            )
            print(f"[judge.system]\n{jpp.system}\n")
            print(f"[judge.user]\n{jpp.user}")
        return 0

    # Per-run log directory under outputs/logs/cot/run_<timestamp>/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.log_root / f"run_{ts}"
    log.info("logging to %s (batch_size=%d)", run_dir, args.batch_size)

    # Heavy deps & remote clients only after we know we're not dry-running.
    model_dir = ensure_model_dir(args.model_id, args.model_cache)
    log.info("model dir: %s", model_dir)
    tokenizer, model = load_model(model_dir, args.dtype, args.device)

    judge_client = None
    if judge_bank is not None:
        judge_client = make_judge_client(judge_bank, args.model_yaml)
        log.info("judge model: %s", judge_client.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.bad_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_csv, \
            open(args.bad_out, "a", encoding="utf-8") as bad_fp, \
            BatchJsonlLogger(run_dir, batch_size=args.batch_size) as blogger:
        for row in tqdm(data, ncols=100, desc="cot"):
            messages = bank.build_messages(
                COT_SCENARIO, {"question": row["question"]}
            )
            raw = generate(messages, tokenizer, model, args.max_new_tokens)
            qwen_answer = extract_answer(raw)
            gold_answer = str(row.get("answer", ""))
            out_csv.write(f"{row['id']},{qwen_answer.replace(chr(10), ' ')}\n")

            entry: dict[str, Any] = {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "qwen_raw": raw,
                "qwen_answer": qwen_answer,
            }

            if judge_bank is not None and judge_client is not None:
                parsed, judge_raw, judge_err = judge_one(
                    judge_bank,
                    judge_client,
                    row["question"],
                    gold_answer,
                    qwen_answer,
                )
                entry["judge"] = parsed
                entry["judge_raw"] = judge_raw
                if judge_err:
                    entry["judge_error"] = judge_err
                if parsed["parsed"] and parsed["correct"] == 0:
                    bad_fp.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "question": row["question"],
                                "gold_answer": gold_answer,
                                "qwen_answer": qwen_answer,
                                "qwen_raw": raw,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    bad_fp.flush()

            blogger.log(entry)

    log.info("wrote %s | logged %d items across %d batches",
             args.output, blogger.total, blogger.batch_id + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
