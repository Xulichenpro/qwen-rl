"""Load YAML prompt config and render Jinja2 templates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jinja2 import Environment, StrictUndefined


@dataclass(frozen=True)
class PromptPair:
    model_key: str
    system: str
    user: str
    temperature: float
    max_tokens: int


class PromptBank:
    def __init__(self, yaml_path: str | Path) -> None:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._raw = raw
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=False,
            lstrip_blocks=False,
        )

    def get_node(self, name: str) -> dict[str, Any]:
        return self._raw[name]

    def render(self, name: str, variables: Mapping[str, Any]) -> PromptPair:
        node = self._raw[name]
        sys_tmpl = self._env.from_string(node["system"])
        usr_tmpl = self._env.from_string(node["user"])
        return PromptPair(
            model_key=node["model_key"],
            system=sys_tmpl.render(**variables),
            user=usr_tmpl.render(**variables),
            temperature=float(node.get("temperature", 0.0)),
            max_tokens=int(node.get("max_tokens", 512)),
        )
