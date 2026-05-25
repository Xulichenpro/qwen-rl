"""Smoke test for the CoT pipeline (prompts + judge + logger).

Runs without a GPU and without loading the Qwen weights or calling Kimi.
Verifies:
1. cot.yml parses and renders into a PromptPair (system, user).
2. cot_judge.yml parses and renders into a PromptPair (judge).
3. extract_answer recovers final answers from <answer> tags.
4. parse_judge_json round-trips a kimi-style JSON object.
5. BatchJsonlLogger rotates every N entries and writes valid JSONL.
6. `run_cot --dry-run` prints both cot and judge rendered prompts.
7. `run_cot --max-items 3 --no-judge --dry-run` honors --max-items.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cot.logger import BatchJsonlLogger  # noqa: E402
from src.cot.parse import extract_answer, parse_judge_json  # noqa: E402
from src.cot.prompts import PromptBank  # noqa: E402

COT_YAML = REPO_ROOT / "configs" / "prompt" / "cot.yml"
JUDGE_YAML = REPO_ROOT / "configs" / "prompt" / "cot_judge.yml"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_cot_render() -> None:
    bank = PromptBank(COT_YAML)
    pp = bank.render("cot", {"question": "9 + 7 = ?"})
    _assert(pp.model_key == "qwen-base", f"unexpected model_key {pp.model_key!r}")
    _assert("<think>" in pp.system, "system missing <think>")
    _assert("<answer>" in pp.system, "system missing <answer>")
    _assert("9 + 7" in pp.user, "user missing new question")
    print("ok  cot_render")


def test_judge_render() -> None:
    bank = PromptBank(JUDGE_YAML)
    node = bank.get_node("judge")
    _assert(node["model_key"] == "kimi-k2.6", f"unexpected judge model_key {node['model_key']!r}")
    pp = bank.render("judge", {"question": "1 + 1 = ?", "qwen_answer": "2"})
    _assert(pp.temperature == 0.0, f"unexpected temperature {pp.temperature}")
    _assert(pp.max_tokens > 0, "judge max_tokens must be positive")
    _assert("题目：1 + 1 = ?" in pp.user, "judge user missing question")
    _assert("Qwen 的答案：2" in pp.user, "judge user missing qwen_answer")
    _assert('"correct"' in pp.system, "judge system missing JSON schema hint")
    print("ok  judge_render")


def test_extract_answer() -> None:
    cases = [
        ("<think>1. ...</think>\n<answer>315</answer>", "315"),
        ("<answer>\n1/2\n</answer>", "1/2"),
        ("<answer>7.5</answer>", "7.5"),
        ("<answer>100</answer><answer>200</answer>", "200"),
        ("no tag, but final 42 .", "42"),
        ("", ""),
        ("纯文字无数字", ""),
    ]
    for text, expected in cases:
        got = extract_answer(text)
        _assert(got == expected, f"extract_answer({text!r}) -> {got!r}, expected {expected!r}")
    print(f"ok  extract_answer: {len(cases)} cases")


def test_parse_judge_json() -> None:
    # Happy path: valid JSON in content.
    parsed = parse_judge_json(
        '{"gold": "315", "qwen": "315", "correct": 1, "reason": "等价"}'
    )
    _assert(parsed["parsed"] is True, "valid JSON should set parsed=True")
    _assert(parsed["correct"] == 1, f"correct=1 expected, got {parsed['correct']}")
    _assert(parsed["gold"] == "315", f"gold field mismatch: {parsed['gold']}")

    # Embedded JSON inside surrounding text.
    parsed = parse_judge_json(
        'noise prefix {"gold": "1/2", "qwen": "0.5", "correct": 0, "reason": "形式不同"} trailing'
    )
    _assert(parsed["parsed"] is True, "embedded JSON should parse")
    _assert(parsed["correct"] == 0, "correct=0 expected")

    # Bad JSON falls back to reasoning channel.
    parsed = parse_judge_json("", '{"correct": 1, "gold": "x", "qwen": "x", "reason": "ok"}')
    _assert(parsed["parsed"] is True, "reasoning JSON fallback failed")

    # Completely unparseable -> parsed=False.
    parsed = parse_judge_json("not json at all", "")
    _assert(parsed["parsed"] is False, "unparseable should set parsed=False")
    _assert(parsed["correct"] == 0, "unparseable should set correct=0")
    print("ok  parse_judge_json")


def test_batch_logger() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run_test"
        with BatchJsonlLogger(run_dir, batch_size=3) as logger:
            for i in range(7):
                logger.log({"i": i, "msg": f"item-{i}"})
        # 7 items, batch_size=3 -> batches 0 (3 lines), 1 (3 lines), 2 (1 line)
        files = sorted(run_dir.glob("batch_*.log"))
        names = [f.name for f in files]
        _assert(
            names == ["batch_0.log", "batch_1.log", "batch_2.log"],
            f"unexpected batch files: {names}",
        )
        line_counts = [sum(1 for _ in open(f, "r", encoding="utf-8")) for f in files]
        _assert(line_counts == [3, 3, 1], f"unexpected line counts: {line_counts}")
        # All lines must be valid JSON.
        for f in files:
            for line in open(f, "r", encoding="utf-8"):
                json.loads(line)
    print("ok  batch_logger")


def test_dry_run_cli_both() -> None:
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.cot.run_cot",
            "--dry-run", "--max-items", "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    _assert(proc.returncode == 0, f"--dry-run exit {proc.returncode}\nstderr:\n{proc.stderr}")
    out = proc.stdout
    _assert("[cot.system]" in out, "missing [cot.system]")
    _assert("[cot.user]" in out, "missing [cot.user]")
    _assert("[judge.system]" in out, "missing [judge.system]")
    _assert("[judge.user]" in out, "missing [judge.user]")
    print("ok  dry_run_cli_both")


def test_dry_run_no_judge() -> None:
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.cot.run_cot",
            "--dry-run", "--no-judge", "--max-items", "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    _assert(proc.returncode == 0, f"--no-judge --dry-run exit {proc.returncode}\nstderr:\n{proc.stderr}")
    out = proc.stdout
    _assert("[cot.system]" in out, "missing [cot.system]")
    _assert("[judge.system]" not in out, "--no-judge should not print judge")
    print("ok  dry_run_no_judge")


def test_max_items_logged() -> None:
    """Confirm --max-items is reflected in the log line."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.cot.run_cot",
            "--dry-run", "--no-judge", "--max-items", "3",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    _assert(proc.returncode == 0, f"exit {proc.returncode}\nstderr:\n{proc.stderr}")
    _assert(
        "items=3" in proc.stderr or "items=3" in proc.stdout,
        f"expected 'items=3' in output\nstderr:\n{proc.stderr}",
    )
    print("ok  max_items_logged")


if __name__ == "__main__":
    test_cot_render()
    test_judge_render()
    test_extract_answer()
    test_parse_judge_json()
    test_batch_logger()
    test_dry_run_cli_both()
    test_dry_run_no_judge()
    test_max_items_logged()
    print("\nALL SMOKE TESTS PASSED")
