import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_build_grpo_dataset_uses_sft_instruction_and_splits_outputs() -> None:
    from src.lora.qwen_ft import INSTRUCTION
    from src.rl.grpo.data import build_grpo_files

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        raw_path = root / "raw.json"
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"

        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": "1",
                        "question": "一加一等于几？",
                        "answer": "2",
                        "instruction": "不要使用这个旧指令",
                    },
                    {
                        "id": "2",
                        "question": "二加二等于几？",
                        "answer": "4",
                        "instruction": "也不要使用这个旧指令",
                    },
                    {
                        "id": "3",
                        "question": "三加三等于几？",
                        "answer": "6",
                    },
                ],
                f,
                ensure_ascii=False,
            )

        stats = build_grpo_files(
            raw_path=raw_path,
            train_path=train_path,
            val_path=val_path,
            max_items=3,
            train_size=2,
            val_size=1,
        )

        assert stats == {
            "raw_total": 3,
            "valid_total": 3,
            "train_total": 2,
            "val_total": 1,
            "skipped_incomplete": 0,
        }

        train_rows = _read_jsonl(train_path)
        val_rows = _read_jsonl(val_path)

        assert train_rows == [
            {
                "id": "1",
                "prompt": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": "一加一等于几？"},
                ],
                "question": "一加一等于几？",
                "gold_answer": "2",
            },
            {
                "id": "2",
                "prompt": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": "二加二等于几？"},
                ],
                "question": "二加二等于几？",
                "gold_answer": "4",
            },
        ]
        assert val_rows == [
            {
                "id": "3",
                "prompt": [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": "三加三等于几？"},
                ],
                "question": "三加三等于几？",
                "gold_answer": "6",
            }
        ]


if __name__ == "__main__":
    test_build_grpo_dataset_uses_sft_instruction_and_splits_outputs()
