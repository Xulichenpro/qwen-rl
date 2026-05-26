import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_build_dpo_dataset_matches_positive_examples_and_splits_outputs() -> None:
    from src.rl.dpo.data import build_dpo_files

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        bad_path = root / "bad.jsonl"
        positive_path = root / "positive.jsonl"
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"

        _write_jsonl(
            bad_path,
            [
                {"id": "1", "question": "一加一等于几？", "qwen_raw": "错解1"},
                {"id": "missing", "question": "没有正例", "qwen_raw": "错解missing"},
                {"id": "2", "question": "二加二等于几？", "qwen_raw": "错解2"},
                {"id": "3", "question": "三加三等于几？", "qwen_raw": "错解3"},
            ],
        )
        _write_jsonl(
            positive_path,
            [
                {"id": "1", "question": "一加一等于几？", "sample": "正解1"},
                {"id": "2", "question": "二加二等于几？", "sample": "正解2"},
                {"id": "3", "question": "三加三等于几？", "sample": "正解3"},
            ],
        )

        stats = build_dpo_files(
            bad_path=bad_path,
            positive_path=positive_path,
            train_path=train_path,
            val_path=val_path,
            max_valid=2,
            train_size=1,
            val_size=1,
        )

        assert stats == {
            "bad_total": 4,
            "positive_total": 3,
            "valid_total": 2,
            "train_total": 1,
            "val_total": 1,
            "skipped_without_positive": 1,
            "skipped_incomplete": 0,
        }

        assert _read_jsonl(train_path) == [
            {
                "prompt": "题目：一加一等于几？\n请逐步推理并给出答案。",
                "chosen": "正解1",
                "rejected": "错解1",
            }
        ]
        assert _read_jsonl(val_path) == [
            {
                "prompt": "题目：二加二等于几？\n请逐步推理并给出答案。",
                "chosen": "正解2",
                "rejected": "错解2",
            }
        ]


if __name__ == "__main__":
    test_build_dpo_dataset_matches_positive_examples_and_splits_outputs()
