"""Optional workflow integration flags (all off by default)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class IntegrationConfig:
    """Which non-Python workflow artifacts to emit."""

    enable_argo: bool = False
    enable_asl: bool = False
    enable_bpmn: bool = False
    enable_python_spec: bool = True

    @property
    def any_enabled(self) -> bool:
        return self.enable_argo or self.enable_asl or self.enable_bpmn


def config_from_env() -> IntegrationConfig:
    return IntegrationConfig(
        enable_argo=_env_bool("ALAS_ENABLE_ARGO"),
        enable_asl=_env_bool("ALAS_ENABLE_ASL"),
        enable_bpmn=_env_bool("ALAS_ENABLE_BPMN"),
        enable_python_spec=_env_bool("ALAS_ENABLE_PYTHON_SPEC", default=True),
    )
