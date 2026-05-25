"""Extract the final numeric answer from a CoT response.

Primary format (per configs/prompt/cot.yml):
    <think>...</think>
    <answer>315</answer>

Fallbacks: bare numeric tail, in case the model drops the tag.
"""
from __future__ import annotations

import re

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_NUMERIC = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Return the model's final answer as a bare string, or '' if not found."""
    if not text:
        return ""
    # Prefer the last <answer>...</answer> block.
    tags = _ANSWER_TAG_RE.findall(text)
    if tags:
        inner = tags[-1].strip()
        nums = _NUMERIC.findall(inner)
        if nums:
            return nums[-1]
        return inner  # tag present but no number — return raw content
    # No tag: fall back to the last numeric token in the whole text.
    nums = _NUMERIC.findall(text)
    return nums[-1] if nums else ""
