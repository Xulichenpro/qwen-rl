"""DPO JSONL dataset helpers.

The project DPO files already use the TRL preference schema:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

``chosen`` is the preferred CoT answer, while ``rejected`` is a Qwen answer
that the judge marked wrong or badly formatted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset


REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}


@dataclass(frozen=True)
class PreferencePreview:
    path: Path
    total: int
    first_prompt: str
    first_chosen: str
    first_rejected: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def load_preference_dataset(path: str | Path, max_items: int = -1) -> Dataset:
    """Load a DPO JSONL file and validate the required preference columns."""
    path = Path(path)
    rows = _read_jsonl(path)
    if max_items >= 0:
        rows = rows[:max_items]

    dataset = Dataset.from_list(rows)
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise KeyError(f"{path} missing DPO columns: {sorted(missing)}")
    return dataset


def preview_preference_file(path: str | Path, max_chars: int = 180) -> PreferencePreview:
    """Return a small preview for dry-run checks without loading any model."""
    path = Path(path)
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")

    first = rows[0]
    missing = REQUIRED_COLUMNS - set(first)
    if missing:
        raise KeyError(f"{path} first row missing columns: {sorted(missing)}")

    def trim(value: Any) -> str:
        text = str(value).replace("\n", "\\n")
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    return PreferencePreview(
        path=path,
        total=len(rows),
        first_prompt=trim(first["prompt"]),
        first_chosen=trim(first["chosen"]),
        first_rejected=trim(first["rejected"]),
    )

