"""Per-run, rotating JSONL logger for the CoT pipeline.

Each `log(obj)` writes a single JSON line to the current `batch_{N}.log`
under the run directory. After `batch_size` entries, a new batch file is
opened so the on-disk layout is::

    outputs/logs/cot/run_<timestamp>/
        batch_0.log    # entries 0..99
        batch_1.log    # entries 100..199
        ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, IO


class BatchJsonlLogger:
    def __init__(self, run_dir: str | Path, batch_size: int = 100) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.run_dir = Path(run_dir)
        self.batch_size = batch_size
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._batch_id = 0
        self._count_in_batch = 0
        self._total = 0
        self._fp: IO[str] | None = self._open(self._batch_id)

    def _open(self, batch_id: int) -> IO[str]:
        path = self.run_dir / f"batch_{batch_id}.log"
        return open(path, "a", encoding="utf-8")

    def log(self, obj: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("logger is closed")
        line = json.dumps(obj, ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()
        self._count_in_batch += 1
        self._total += 1
        if self._count_in_batch >= self.batch_size:
            self._fp.close()
            self._batch_id += 1
            self._count_in_batch = 0
            self._fp = self._open(self._batch_id)

    @property
    def total(self) -> int:
        return self._total

    @property
    def batch_id(self) -> int:
        return self._batch_id

    def close(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.close()
        self._fp = None

    def __enter__(self) -> "BatchJsonlLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
