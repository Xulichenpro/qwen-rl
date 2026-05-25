"""LangChain ChatOpenAI wrappers for the in-house vLLM endpoints.

Reads configs/model/models.yml (gitignored), which holds base_url + api_key
per model. Per-call inference params (temperature, max_tokens) are passed
into ChatClient at construction time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass(frozen=True)
class ChatReply:
    content: str
    reasoning: str

    def merged(self) -> str:
        """If content is empty, fall back to reasoning; else return content."""
        return self.content if self.content.strip() else self.reasoning


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    model: str
    base_url: str
    api_key: str
    timeout: int
    max_retries: int


def load_endpoints(yaml_path: str | Path) -> dict[str, ModelEndpoint]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out: dict[str, ModelEndpoint] = {}
    for key, cfg in raw.items():
        out[key] = ModelEndpoint(
            name=key,
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=int(cfg.get("timeout", 60)),
            max_retries=int(cfg.get("max_retries", 2)),
        )
    return out


class ChatClient:
    """Thread-safe wrapper around ChatOpenAI with tenacity retry."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._ep = endpoint
        self._llm = ChatOpenAI(
            model=endpoint.model,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=endpoint.timeout,
            max_retries=0,  # we wrap our own retry below
        )

    @property
    def model(self) -> str:
        return self._ep.model

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def chat(self, system: str, user: str) -> ChatReply:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = self._llm.invoke(msgs)
        content = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )
        reasoning = _extract_reasoning(resp)
        return ChatReply(content=content, reasoning=reasoning)


def _extract_reasoning(resp: Any) -> str:
    """Best-effort extraction of `reasoning_content` from various langchain
    / openai response shapes.
    """
    # 1) additional_kwargs (most common with reasoning-capable OAI-compatible APIs)
    ak = getattr(resp, "additional_kwargs", None) or {}
    for k in ("reasoning_content", "reasoning"):
        v = ak.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for sub in ("content", "text", "summary"):
                sv = v.get(sub)
                if isinstance(sv, str) and sv.strip():
                    return sv

    # 2) response_metadata.message_extras / .reasoning
    md = getattr(resp, "response_metadata", None) or {}
    for k in ("reasoning_content", "reasoning"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v

    return ""
