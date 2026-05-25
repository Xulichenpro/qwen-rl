"""Per-item generate -> judge pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .clients import ChatClient
from .prompts import PromptBank


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"</?tool_call>", re.IGNORECASE)


@dataclass
class SynthResult:
    id: str
    question: str
    gold_answer: str
    sample: str
    think: str
    model_answer: str
    judge_raw: str
    correct: int
    concise: int
    passed: int
    reason: str
    generator_model: str
    judge_model: str
    error: str = ""


def _build_sample(content: str, reasoning: str) -> tuple[str, str, str]:
    """Return (canonical_sample, think, answer).

    The reasoning model often puts CoT in `reasoning_content` and emits only
    `<answer>...</answer>` (or unbalanced `</think>`) in `content`. We
    reconstruct a canonical `<think>...</think><answer>...</answer>` string.
    """
    content = _TOOL_CALL_RE.sub("", content).strip()
    reasoning = reasoning.strip()

    # answer block: look in content first, then in reasoning as fallback
    a = _ANSWER_RE.search(content) or _ANSWER_RE.search(reasoning)
    answer = a.group(1).strip() if a else ""

    # think text: prefer explicit <think>...</think>; else use reasoning;
    # else use content minus the <answer> block.
    t = _THINK_RE.search(content) or _THINK_RE.search(reasoning)
    if t:
        think = t.group(1).strip()
    elif reasoning:
        think = reasoning
    else:
        # strip <answer>...</answer> and any stray </think> from content
        stripped = _ANSWER_RE.sub("", content)
        stripped = re.sub(r"</?think>", "", stripped, flags=re.IGNORECASE)
        think = stripped.strip()

    sample = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
    return sample, think, answer


def _try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_judge(content: str, reasoning: str = "") -> dict[str, Any]:
    obj = _try_parse_json(content)
    if obj is None:
        obj = _try_parse_json(reasoning)
    if obj is None:
        had_text = bool((content or "").strip() or (reasoning or "").strip())
        return {
            "correct": 0,
            "concise": 0,
            "pass": 0,
            "reason": "judge: bad json" if had_text else "judge: no json",
        }
    return {
        "correct": int(obj.get("correct", 0)),
        "concise": int(obj.get("concise", 0)),
        "pass": int(obj.get("pass", 0)),
        "reason": str(obj.get("reason", ""))[:80],
    }


def synthesize_one(
    item: dict[str, Any],
    bank: PromptBank,
    generator: ChatClient,
    judge: ChatClient,
) -> SynthResult:
    qid = str(item["id"])
    question = item["question"]
    gold = str(item["answer"])

    try:
        gen_pp = bank.render(
            "generator", {"question": question, "answer": gold}
        )
        gen_reply = generator.chat(gen_pp.system, gen_pp.user)
        sample, think, model_ans = _build_sample(
            gen_reply.content, gen_reply.reasoning
        )

        jud_pp = bank.render(
            "judge",
            {"question": question, "answer": gold, "sample": sample},
        )
        jud_reply = judge.chat(jud_pp.system, jud_pp.user)
        judge_raw = jud_reply.merged()
        parsed = _parse_judge(jud_reply.content, jud_reply.reasoning)
    except Exception as exc:  # noqa: BLE001
        return SynthResult(
            id=qid,
            question=question,
            gold_answer=gold,
            sample="",
            think="",
            model_answer="",
            judge_raw="",
            correct=0,
            concise=0,
            passed=0,
            reason="",
            generator_model=generator.model,
            judge_model=judge.model,
            error=f"{type(exc).__name__}: {exc}",
        )

    return SynthResult(
        id=qid,
        question=question,
        gold_answer=gold,
        sample=sample,
        think=think,
        model_answer=model_ans,
        judge_raw=judge_raw,
        correct=parsed["correct"],
        concise=parsed["concise"],
        passed=parsed["pass"],
        reason=parsed["reason"],
        generator_model=generator.model,
        judge_model=judge.model,
    )
