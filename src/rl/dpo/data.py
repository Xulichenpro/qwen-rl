"""Build DPO train/validation JSONL files from positive and rejected samples.

The builder keeps the order from `bad_out.jsonl`, takes the first 2,500 rows
that have a matching positive CoT in `train_cot.jsonl`, and writes the DPO
schema expected by TRL-style trainers:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python -m src.rl.dpo.data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BAD_PATH = REPO_ROOT / "datasets" / "dpo_train" / "bad_out.jsonl"
DEFAULT_POSITIVE_PATH = REPO_ROOT / "datasets" / "syn_train" / "train_cot.jsonl"
DEFAULT_TRAIN_PATH = REPO_ROOT / "datasets" / "dpo_train" / "train.jsonl"
DEFAULT_VAL_PATH = REPO_ROOT / "datasets" / "dpo_train" / "val.jsonl"
DEFAULT_MAX_VALID = 2500
DEFAULT_TRAIN_SIZE = 2000
DEFAULT_VAL_SIZE = 500
PROMPT_TEMPLATE = "题目：{question}\n请逐步推理并给出答案。"


def _load_jsonl(path: Path):
    """Load a JSONL file through Hugging Face datasets."""
    return load_dataset("json", data_files=str(path), split="train")


def _non_empty_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_positive_by_id(positive_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    positive_dataset = _load_jsonl(positive_path)
    positive_by_id: dict[str, dict[str, str]] = {}

    for row in positive_dataset:
        row_id = _non_empty_text(row, "id")
        question = _non_empty_text(row, "question")
        chosen = _non_empty_text(row, "sample")
        if row_id is None or question is None or chosen is None:
            continue
        positive_by_id.setdefault(
            row_id,
            {
                "question": question,
                "chosen": chosen,
            },
        )

    return positive_by_id, len(positive_dataset)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dpo_files(
    *,
    bad_path: Path = DEFAULT_BAD_PATH,
    positive_path: Path = DEFAULT_POSITIVE_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    max_valid: int = DEFAULT_MAX_VALID,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict[str, int]:
    """Create DPO train/validation JSONL files and return build statistics."""
    if max_valid <= 0:
        raise ValueError("max_valid must be positive")
    if train_size < 0 or val_size < 0:
        raise ValueError("train_size and val_size must be non-negative")
    if train_size + val_size > max_valid:
        raise ValueError("train_size + val_size cannot exceed max_valid")

    positive_by_id, positive_total = _build_positive_by_id(positive_path)
    bad_dataset = _load_jsonl(bad_path)

    samples: list[dict[str, str]] = []
    skipped_without_positive = 0
    skipped_incomplete = 0

    for row in bad_dataset:
        if len(samples) >= max_valid:
            break

        row_id = _non_empty_text(row, "id")
        question = _non_empty_text(row, "question")
        rejected = _non_empty_text(row, "qwen_raw")
        if row_id is None or question is None or rejected is None:
            skipped_incomplete += 1
            continue

        positive = positive_by_id.get(row_id)
        if positive is None:
            skipped_without_positive += 1
            continue

        samples.append(
            {
                "prompt": PROMPT_TEMPLATE.format(question=question),
                "chosen": positive["chosen"],
                "rejected": rejected,
            }
        )

    required_total = train_size + val_size
    if len(samples) < required_total:
        raise ValueError(
            f"only found {len(samples)} valid samples, but "
            f"{required_total} are required"
        )

    train_rows = samples[:train_size]
    val_rows = samples[train_size:required_total]

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    return {
        "bad_total": len(bad_dataset),
        "positive_total": positive_total,
        "valid_total": required_total,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "skipped_without_positive": skipped_without_positive,
        "skipped_incomplete": skipped_incomplete,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad-path", type=Path, default=DEFAULT_BAD_PATH)
    parser.add_argument("--positive-path", type=Path, default=DEFAULT_POSITIVE_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--max-valid", type=int, default=DEFAULT_MAX_VALID)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_dpo_files(
        bad_path=args.bad_path,
        positive_path=args.positive_path,
        train_path=args.train_path,
        val_path=args.val_path,
        max_valid=args.max_valid,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
