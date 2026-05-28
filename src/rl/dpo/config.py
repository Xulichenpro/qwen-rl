"""Configuration loader for the readable math DPO pipeline.

This module intentionally mirrors ``configs/train/dpo.yml`` instead of
inventing a second config format. The YAML is rendered with Jinja2 first, so
fields such as ``model.local_dir`` can reference ``model.id``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined


REQUIRED_SECTIONS = ("model", "data", "training", "swanlab")


@dataclass(frozen=True)
class MathDpoConfig:
    model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    swanlab: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MathDpoConfig":
        missing = [section for section in REQUIRED_SECTIONS if section not in raw]
        if missing:
            raise KeyError(f"dpo config missing sections: {missing}")
        return cls(
            model=raw["model"],
            data=raw["data"],
            training=raw["training"],
            swanlab=raw["swanlab"],
        )


def _render_yaml(text: str, *, max_passes: int = 4) -> dict[str, Any]:
    """Render Jinja2 expressions in YAML until the text stops changing."""
    env = Environment(undefined=StrictUndefined, autoescape=False)
    current = text
    parsed: dict[str, Any] = yaml.safe_load(current) or {}

    for _ in range(max_passes):
        rendered = env.from_string(current).render(**parsed)
        if rendered == current:
            break
        current = rendered
        parsed = yaml.safe_load(current) or {}

    return parsed


def load_config(path: str | Path) -> MathDpoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = _render_yaml(f.read())
    return MathDpoConfig.from_dict(raw)

