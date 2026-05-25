"""Load YAML CoT prompt config and render Jinja2 templates.

Mirrors the layout used by src/data_syn/prompts.py:
- top-level scenario keys (here: `cot`)
- each scenario has `model_key`, `system`, `user`
Few-shot demonstrations are inlined into the `system` block of cot.yml.
"""
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

    def render(self, name: str, variables: Mapping[str, Any]) -> PromptPair:
        node = self._raw[name]
        sys_tmpl = self._env.from_string(node["system"])
        usr_tmpl = self._env.from_string(node["user"])
        return PromptPair(
            model_key=node["model_key"],
            system=sys_tmpl.render(**variables),
            user=usr_tmpl.render(**variables),
        )

    def build_messages(
        self, name: str, variables: Mapping[str, Any]
    ) -> list[dict[str, str]]:
        """Render a scenario into a chat-message list for apply_chat_template."""
        pp = self.render(name, variables)
        return [
            {"role": "system", "content": pp.system},
            {"role": "user", "content": pp.user},
        ]
