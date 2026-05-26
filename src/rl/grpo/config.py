"""Load GRPO training config from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

_REQUIRED_SECTIONS = ("model", "data", "training", "swanlab")


@dataclass(frozen=True)
class GrpoConfig:
    model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    swanlab: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GrpoConfig":
        missing = [section for section in _REQUIRED_SECTIONS if section not in raw]
        if missing:
            raise KeyError(f"grpo config missing sections: {missing}")
        return cls(
            model=raw["model"],
            data=raw["data"],
            training=raw["training"],
            swanlab=raw["swanlab"],
        )


def _render_yaml(text: str, *, max_passes: int = 4) -> dict[str, Any]:
    env = Environment(undefined=StrictUndefined, autoescape=False)
    current = text
    last_parsed: dict[str, Any] = yaml.safe_load(current) or {}
    for _ in range(max_passes):
        rendered = env.from_string(current).render(**last_parsed)
        if rendered == current:
            break
        current = rendered
        last_parsed = yaml.safe_load(current) or {}
    return last_parsed


def load_grpo_config(path: str | Path) -> GrpoConfig:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return GrpoConfig.from_dict(_render_yaml(text))
