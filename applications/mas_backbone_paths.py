"""Output paths and source tags for P4/P10 MAS batches."""

from __future__ import annotations

import os


def normalize_backbone(name: str) -> str:
    n = name.strip().lower()
    if n in ("gpt-4o", "gpt4o", "openai"):
        return "gpt-4o"
    return "claude-4"


def mas_output_dir(project_root: str, backbone: str) -> str:
    return os.path.join(project_root, f"results_mas({normalize_backbone(backbone)})")


def mas_source_tag(backbone: str) -> str:
    return "MAS-GPT4o" if normalize_backbone(backbone) == "gpt-4o" else "MAS-Claude4"
