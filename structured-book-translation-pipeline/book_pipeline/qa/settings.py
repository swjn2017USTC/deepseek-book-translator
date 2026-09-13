"""Tolerant ``qa{}`` config parsing (plan §D5).

An absent or partial ``config["qa"]`` block must never break the deterministic
cascade: unknown keys are ignored and every supported key has a documented
default so legacy P03-P06 configs run QA-1/2/3 unchanged and QA-4 only when
``enabled`` and the provider probe passes.

Parsing intentionally does NOT share the naming/validation style of
``contextual.orchestrator._v2_settings`` beyond the values below; the QA block
is a purely additive consumer of an optional config key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class QaSettings:
    """Resolved QA thresholds (spec §13 / plan §3.3 + §D5 defaults)."""

    enabled: bool = True
    max_len_ratio: float = 0.5       # target/src char ratio floor -> suspiciously short
    max_residual_run: int = 8        # consecutive source-script words -> residual-run flag
    critic_role: str = "qa_critic"   # llm_roles key (resolve_model falls back to provider.model)
    wire_budget: int = 20            # max critic HTTP attempts per qa run


_KNOWN = {
    "enabled": bool,
    "max_len_ratio": float,
    "max_residual_run": int,
    "critic_role": str,
    "wire_budget": int,
}


def qa_settings(config: Dict[str, Any]) -> QaSettings:
    """Parse ``config["qa"]`` tolerantly; unknown keys are ignored, missing
    keys keep the documented defaults (plan §D5)."""
    block = config.get("qa")
    if not isinstance(block, dict):
        return QaSettings()
    values: Dict[str, Any] = {}
    for key in _KNOWN:
        if key in block:
            values[key] = block[key]
    # Validate ranges; a broken value is a loud config error, never a silent
    # fallback to defaults (the book owner asked for the value).
    if values.get("max_len_ratio") is not None and not 0.0 < float(values["max_len_ratio"]) <= 1.0:
        raise ValueError("qa.max_len_ratio must be in (0.0, 1.0]")
    for key in ("max_residual_run", "wire_budget"):
        if values.get(key) is not None and int(values[key]) < 0:
            raise ValueError(f"qa.{key} must be non-negative")
    return QaSettings(**values)
