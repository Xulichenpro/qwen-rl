"""Build GRPO train/validation JSONL files from raw math data.

The GRPO prompt intentionally reuses the SFT system instruction so rollout
training optimizes the same response contract as supervised fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.lora.qwen_ft import INSTRUCTION

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_PATH = REPO_ROOT / "datasets" / "raw_train" / "train.json"
DEFAULT_TRAIN_PATH = REPO_ROOT / "datasets" / "grpo_train" / "train.jsonl"
DEFAULT_VAL_PATH = REPO_ROOT / "datasets" / "grpo_train" / "val.jsonl"
DEFAULT_MAX_ITEMS = 2500
DEFAULT_TRAIN_SIZE = 2000
DEFAULT_VAL_SIZE = 500


def _non_empty_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_raw_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_grpo_row(row: dict[str, Any]) -> dict[str, Any] | None:
    row_id = _non_empty_text(row, "id")
    question = _non_empty_text(row, "question")
    gold_answer = _non_empty_text(row, "answer")
    if row_id is None or question is None or gold_answer is None:
        return None
    return {
        "id": row_id,
        "prompt": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": question},
        ],
        "question": question,
        "gold_answer": gold_answer,
    }


def build_grpo_files(
    *,
    raw_path: Path = DEFAULT_RAW_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    max_items: int = DEFAULT_MAX_ITEMS,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict[str, int]:
    """Create GRPO train/validation JSONL files and return build statistics."""
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if train_size < 0 or val_size < 0:
        raise ValueError("train_size and val_size must be non-negative")
    if train_size + val_size > max_items:
        raise ValueError("train_size + val_size cannot exceed max_items")

    raw_rows = _load_raw_json(raw_path)
    grpo_rows: list[dict[str, Any]] = []
    skipped_incomplete = 0

    for row in raw_rows:
        if len(grpo_rows) >= max_items:
            break
        grpo_row = _to_grpo_row(row)
        if grpo_row is None:
            skipped_incomplete += 1
            continue
        grpo_rows.append(grpo_row)

    required_total = train_size + val_size
    if len(grpo_rows) < required_total:
        raise ValueError(
            f"only found {len(grpo_rows)} valid samples, but "
            f"{required_total} are required"
        )

    train_rows = grpo_rows[:train_size]
    val_rows = grpo_rows[train_size:required_total]
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    return {
        "raw_total": len(raw_rows),
        "valid_total": required_total,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "skipped_incomplete": skipped_incomplete,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_grpo_files(
        raw_path=args.raw_path,
        train_path=args.train_path,
        val_path=args.val_path,
        max_items=args.max_items,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
