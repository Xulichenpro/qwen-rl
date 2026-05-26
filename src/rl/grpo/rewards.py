"""Reward functions for GRPO math training."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^-?\d+(?:/\d+)?(?:\.\d+)?$")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def _extract_answer(text: str) -> str:
    matches = _ANSWER_TAG_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def _to_fraction(text: str) -> Fraction | None:
    normalized = text.strip().replace("，", ",")
    if not normalized:
        return None
    try:
        if "/" in normalized:
            return Fraction(normalized)
        return Fraction(Decimal(normalized))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def _answers_equal(predicted: str, gold: str) -> bool:
    pred_fraction = _to_fraction(predicted)
    gold_fraction = _to_fraction(gold)
    if pred_fraction is not None and gold_fraction is not None:
        return pred_fraction == gold_fraction
    return predicted.strip() == gold.strip()


def answer_reward(completions: list[Any], gold_answer: list[str], **kwargs) -> list[float]:
    """Reward exact final-answer correctness after numeric normalization."""
    rewards: list[float] = []
    for completion, gold in zip(completions, gold_answer):
        predicted = _extract_answer(_completion_text(completion))
        rewards.append(1.0 if predicted and _answers_equal(predicted, str(gold)) else 0.0)
    return rewards


def format_reward(completions: list[Any], **kwargs) -> list[float]:
    """Reward the SFT output contract: one think block and one bare answer."""
    rewards: list[float] = []
    for completion in completions:
        text = _completion_text(completion).strip()
        think_matches = _THINK_TAG_RE.findall(text)
        answer_matches = _ANSWER_TAG_RE.findall(text)
        if len(think_matches) != 1 or len(answer_matches) != 1:
            rewards.append(0.0)
            continue
        answer = answer_matches[0].strip()
        rewards.append(1.0 if _BARE_NUMBER_RE.fullmatch(answer) else 0.0)
    return rewards


def concise_reward(completions: list[Any], **kwargs) -> list[float]:
    """Small reward for keeping the reasoning within the SFT instruction limit."""
    rewards: list[float] = []
    for completion in completions:
        text = _completion_text(completion)
        think_matches = _THINK_TAG_RE.findall(text)
        if len(think_matches) != 1:
            rewards.append(0.0)
            continue
        non_empty_lines = [
            line for line in think_matches[0].splitlines() if line.strip()
        ]
        rewards.append(1.0 if 1 <= len(non_empty_lines) <= 6 else 0.0)
    return rewards
