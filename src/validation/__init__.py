"""
src/validation/__init__.py
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_quality import (
        CheckResult,
        DataQualityReport,
        check_categorical_domains,
        check_critical_nulls,
        check_numeric_ranges,
        check_schema,
        check_target_labels,
        check_temporal_span,
        check_uniqueness,
        run_data_quality_suite,
    )


def __getattr__(name: str):
    if name in {
        "CheckResult",
        "DataQualityReport",
        "check_schema",
        "check_uniqueness",
        "check_target_labels",
        "check_critical_nulls",
        "check_numeric_ranges",
        "check_categorical_domains",
        "check_temporal_span",
        "run_data_quality_suite",
    }:
        import src.validation.data_quality as dq
        return getattr(dq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CheckResult",
    "DataQualityReport",
    "check_schema",
    "check_uniqueness",
    "check_target_labels",
    "check_critical_nulls",
    "check_numeric_ranges",
    "check_categorical_domains",
    "check_temporal_span",
    "run_data_quality_suite",
]
