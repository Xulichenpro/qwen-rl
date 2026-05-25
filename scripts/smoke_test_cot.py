"""Smoke test for the CoT pipeline.

Runs without a GPU and without loading the Qwen weights. Verifies:
1. cot.yml parses and renders into a (system, user) PromptPair.
2. build_messages returns a well-formed [system, user] chat list.
3. parse.extract_answer recovers final answers from <answer>...</answer> outputs.
4. `run_cot --dry-run` works end-to-end on the real test.json.

Exit status: 0 on success, 1 on first failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cot.parse import extract_answer  # noqa: E402
from src.cot.prompts import PromptBank  # noqa: E402

YAML_PATH = REPO_ROOT / "configs" / "prompt" / "cot.yml"
SCENARIO = "cot"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_prompt_render() -> None:
    bank = PromptBank(YAML_PATH)
    pp = bank.render(SCENARIO, {"question": "9 + 7 = ?"})
    _assert(pp.model_key == "qwen-base", f"unexpected model_key {pp.model_key!r}")
    _assert(bool(pp.system.strip()), "system prompt should be non-empty")
    _assert("<think>" in pp.system, "system prompt should mention <think>")
    _assert("<answer>" in pp.system, "system prompt should mention <answer>")
    _assert(
        "题目：食堂运来105千克" in pp.system,
        "system prompt should embed the few-shot demos",
    )
    _assert("9 + 7" in pp.user, "user message should carry the new question")
    print("ok  prompt_render")


def test_build_messages() -> None:
    bank = PromptBank(YAML_PATH)
    msgs = bank.build_messages(SCENARIO, {"question": "1 + 1 = ?"})
    _assert(len(msgs) == 2, f"expected 2 messages, got {len(msgs)}")
    _assert(msgs[0]["role"] == "system", "first message must be system")
    _assert(msgs[1]["role"] == "user", "second message must be user")
    _assert("1 + 1" in msgs[1]["content"], "user message missing new question")
    print("ok  build_messages")


def test_strict_undefined() -> None:
    """An unsupplied variable in user template should raise UndefinedError."""
    from jinja2 import UndefinedError

    bank = PromptBank(YAML_PATH)
    try:
        bank.render(SCENARIO, {})  # `question` missing
    except UndefinedError:
        print("ok  strict_undefined")
        return
    _assert(False, "expected UndefinedError when 'question' is missing")


def test_extract_answer() -> None:
    cases = [
        ("<think>1. ...\n2. ...</think>\n<answer>315</answer>", "315"),
        ("<think>x</think><answer>\n1/2\n</answer>", "1/2"),
        ("<answer>7.5</answer>", "7.5"),
        ("<answer>第一答 100</answer><answer>200</answer>", "200"),  # last tag wins
        ("no tags but final number 42 .", "42"),
        ("", ""),
        ("纯文字无数字", ""),
    ]
    for text, expected in cases:
        got = extract_answer(text)
        _assert(
            got == expected,
            f"extract_answer({text!r}) -> {got!r}, expected {expected!r}",
        )
    print(f"ok  extract_answer: {len(cases)} cases")


def test_dry_run_cli() -> None:
    """Invoke run_cot with --dry-run; verify it prints a rendered prompt."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cot.run_cot",
            "--dry-run",
            "--limit",
            "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    _assert(
        proc.returncode == 0,
        f"--dry-run exit {proc.returncode}\nstderr:\n{proc.stderr}",
    )
    out = proc.stdout
    _assert("[system]" in out, "--dry-run stdout missing [system]")
    _assert("[user]" in out, "--dry-run stdout missing [user]")
    _assert("题目：" in out, "--dry-run stdout missing 题目：")
    _assert("<think>" in out, "--dry-run stdout missing <think>")
    print("ok  dry_run_cli")


if __name__ == "__main__":
    test_prompt_render()
    test_build_messages()
    test_strict_undefined()
    test_extract_answer()
    test_dry_run_cli()
    print("\nALL SMOKE TESTS PASSED")
