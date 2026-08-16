"""
src/models/__init__.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluation import (
        bootstrap_metric_confidence_intervals,
        calculate_recall_at_fixed_fpr,
        compute_classification_metrics,
        generate_threshold_sweep,
    )
    from .train import run_training_pipeline


def __getattr__(name: str):
    if name in {
        "bootstrap_metric_confidence_intervals",
        "calculate_recall_at_fixed_fpr",
        "compute_classification_metrics",
        "generate_threshold_sweep",
    }:
        import src.models.evaluation as ev
        return getattr(ev, name)
    elif name == "run_training_pipeline":
        import src.models.train as tr
        return getattr(tr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_classification_metrics",
    "calculate_recall_at_fixed_fpr",
    "bootstrap_metric_confidence_intervals",
    "generate_threshold_sweep",
    "run_training_pipeline",
]
