"""
src/eda/__init__.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .insights import (
        calculate_risk_ratio,
        compute_all_eda_insights,
        wilson_confidence_interval,
    )


def __getattr__(name: str):
    if name in {"calculate_risk_ratio", "compute_all_eda_insights", "wilson_confidence_interval"}:
        import src.eda.insights as ins
        return getattr(ins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "wilson_confidence_interval",
    "calculate_risk_ratio",
    "compute_all_eda_insights",
]
