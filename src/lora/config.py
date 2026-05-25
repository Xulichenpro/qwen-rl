"""Load LoRA train/inference config from YAML.

The YAML at `configs/train/lora.yml` is first parsed by PyYAML and then
re-rendered through Jinja2 with the parsed dict as context. This lets any
string field reference other sections, e.g.
    local_dir: "{{ model.cache_dir }}{{ model.id }}/"

`load_lora_config(path)` returns a `LoraConfig` dataclass with five sections:
    .model .data .lora .training .swanlab .inference
Each section is a plain dict; lookup is by `cfg.section["key"]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined


_REQUIRED_SECTIONS = ("model", "data", "lora", "training", "swanlab", "inference")


@dataclass(frozen=True)
class LoraConfig:
    model: dict[str, Any]
    data: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]
    swanlab: dict[str, Any]
    inference: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LoraConfig":
        missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
        if missing:
            raise KeyError(f"lora config missing sections: {missing}")
        return cls(
            model=raw["model"],
            data=raw["data"],
            lora=raw["lora"],
            training=raw["training"],
            swanlab=raw["swanlab"],
            inference=raw["inference"],
        )


def _render_yaml(text: str, *, max_passes: int = 4) -> dict[str, Any]:
    """Iteratively render the YAML text through Jinja2 until it converges.

    Multiple passes let derived fields reference other derived fields, e.g.
    `swanlab.experiment_name` referencing `model.id`.
    """
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


def load_lora_config(path: str | Path) -> LoraConfig:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    raw = _render_yaml(text)
    return LoraConfig.from_dict(raw)


def resolve_torch_dtype(name: str):
    """Map a YAML dtype string to a torch dtype. Imported lazily."""
    import torch

    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unsupported torch_dtype: {name!r}")
    return table[name]


def resolve_task_type(name: str):
    """Map a YAML task_type string to peft.TaskType. Imported lazily."""
    from peft import TaskType

    try:
        return TaskType[name]
    except KeyError as exc:
        raise ValueError(f"unsupported peft TaskType: {name!r}") from exc
