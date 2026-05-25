"""Extract Qwen final answers and parse Kimi judge JSON."""
from __future__ import annotations

import json
import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_NUMERIC = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_answer(text: str) -> str:
    """Return Qwen's final answer as a bare string, or '' if not found.

    Primary: last <answer>...</answer> block in the response. If the tag is
    present but holds non-numeric content, return the inner text as-is.
    Fallback: last numeric token anywhere in the text.
    """
    if not text:
        return ""
    tags = _ANSWER_TAG_RE.findall(text)
    if tags:
        inner = tags[-1].strip()
        nums = _NUMERIC.findall(inner)
        if nums:
            return nums[-1]
        return inner
    nums = _NUMERIC.findall(text)
    return nums[-1] if nums else ""


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


def parse_judge_json(content: str, reasoning: str = "") -> dict[str, Any]:
    """Parse Kimi's judge response into a structured dict.

    Returns a dict with these keys (always present):
        parsed:  True if a JSON object was recovered, else False
        gold:    judge-computed answer (str)
        qwen:    qwen's answer as the judge echoed it (str)
        correct: 0 or 1
        reason:  judge's short rationale, truncated to 80 chars

    On unparseable input, `parsed=False, correct=0, reason="judge: bad json"`.
    """
    obj = _try_parse_json(content)
    if obj is None:
        obj = _try_parse_json(reasoning)
    if obj is None:
        had_text = bool((content or "").strip() or (reasoning or "").strip())
        return {
            "parsed": False,
            "gold": "",
            "qwen": "",
            "correct": 0,
            "reason": "judge: bad json" if had_text else "judge: no json",
        }
    return {
        "parsed": True,
        "gold": str(obj.get("gold", "")),
        "qwen": str(obj.get("qwen", "")),
        "correct": int(obj.get("correct", 0)),
        "reason": str(obj.get("reason", ""))[:80],
    }
