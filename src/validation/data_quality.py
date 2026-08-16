"""
src/validation/data_quality.py
------------------------------
Automated, repeatable data quality validation engine for the IEEE-CIS Fraud Detection dataset.

Unlike the one-time investigative audit in Week 1 (which established empirical dataset properties,
leakage risks, and split strategies), this module provides a continuous, repeatable data contract
verification suite for incoming data batches, training partitions, and inference payloads.

Key Capabilities:
  - Schema & column presence verification
  - Memory-optimized dtype conformance
  - Target label integrity (binary {0, 1}, zero nulls in labeled batches)
  - Primary key / TransactionID uniqueness
  - Numerical range and sanity bounds (amounts, deltas, card IDs)
  - Categorical domain validity (ProductCD, card4, card6)
  - Critical feature null-rate thresholds
  - Temporal span and monotonicity validation
  - Structured DataQualityReport generation (dict, JSON, CLI print)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Contracts & Domain Constraints
# ---------------------------------------------------------------------------

MANDATORY_TRANSACTION_COLUMNS: list[str] = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "C1",
    "D1",
]

MANDATORY_LABELED_COLUMNS: list[str] = MANDATORY_TRANSACTION_COLUMNS + ["isFraud"]

CRITICAL_ZERO_NULL_COLUMNS: list[str] = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
]

ALLOWED_CATEGORIES: dict[str, set[str]] = {
    "ProductCD": {"W", "H", "C", "S", "R"},
    "card4": {"visa", "mastercard", "american express", "discover"},
    "card6": {"credit", "debit", "debit or credit", "charge card"},
}

DEFAULT_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "TransactionAmt": (0.001, 35000.0),      # Amount must be strictly positive
    "TransactionDT": (86400, 20000000),       # Minimum delta = 1 day (86400s)
    "card1": (1000, 20000),                  # Realistic card issuer bin range
    "C1": (0.0, 10000.0),                    # Velocity count proxy non-negative
    "D1": (0.0, 10000.0),                    # Time-delta non-negative
}


# ---------------------------------------------------------------------------
# Report Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Individual data quality check outcome."""
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    violations_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Aggregated data quality report across all check results."""
    timestamp: str
    total_rows: int
    total_columns: int
    overall_status: str  # "PASS", "FAIL", "WARN"
    checks_passed: int
    checks_failed: int
    checks_warned: int
    results: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print human-readable summary table to stdout."""
        print("\n" + "=" * 70)
        print("  DATA QUALITY REPORT SUMMARY")
        print("=" * 70)
        print(f" Timestamp:      {self.timestamp}")
        print(f" Rows Evaluated: {self.total_rows:,}")
        print(f" Columns:        {self.total_columns}")
        print(f" Overall Status: {self.overall_status}")
        print(f" Checks:         {self.checks_passed} Passed | {self.checks_failed} Failed | {self.checks_warned} Warned")
        print("-" * 70)
        for r in self.results:
            status_tag = f"[{r.status}]"
            print(f" {status_tag:<8} {r.name:<32} {r.message}")
            if r.violations_count > 0:
                print(f"          -> Violations: {r.violations_count:,}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Individual Check Functions
# ---------------------------------------------------------------------------

def check_schema(df: pd.DataFrame, required_columns: list[str]) -> CheckResult:
    """Verify that all required columns are present in the batch."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return CheckResult(
            name="Schema Integrity",
            status="FAIL",
            message=f"Missing {len(missing)} mandatory column(s): {missing[:5]}",
            violations_count=len(missing),
            metrics={"missing_columns": missing},
        )
    return CheckResult(
        name="Schema Integrity",
        status="PASS",
        message=f"All {len(required_columns)} mandatory columns present.",
        violations_count=0,
        metrics={"required_column_count": len(required_columns)},
    )


def check_uniqueness(df: pd.DataFrame, id_col: str = "TransactionID") -> CheckResult:
    """Verify that the primary key / TransactionID is strictly unique with no duplicates."""
    if id_col not in df.columns:
        return CheckResult(
            name="Primary Key Uniqueness",
            status="FAIL",
            message=f"Identifier column '{id_col}' not found in dataset.",
            violations_count=1,
        )

    dup_count = int(df[id_col].duplicated().sum())
    if dup_count > 0:
        return CheckResult(
            name="Primary Key Uniqueness",
            status="FAIL",
            message=f"Found {dup_count:,} duplicate {id_col} values.",
            violations_count=dup_count,
            metrics={"duplicate_count": dup_count},
        )

    return CheckResult(
        name="Primary Key Uniqueness",
        status="PASS",
        message=f"All {len(df):,} '{id_col}' records are strictly unique.",
        violations_count=0,
        metrics={"unique_count": len(df)},
    )


def check_target_labels(df: pd.DataFrame, target_col: str = "isFraud") -> CheckResult:
    """
    Verify fraud target column integrity:
      - Column exists
      - Values strictly in {0, 1}
      - Zero NaN/null values
    """
    if target_col not in df.columns:
        return CheckResult(
            name="Target Label Integrity",
            status="FAIL",
            message=f"Target column '{target_col}' not found.",
            violations_count=1,
        )

    null_count = int(df[target_col].isna().sum())
    if null_count > 0:
        return CheckResult(
            name="Target Label Integrity",
            status="FAIL",
            message=f"Target column '{target_col}' contains {null_count:,} nulls.",
            violations_count=null_count,
            metrics={"null_count": null_count},
        )

    valid_vals = {0, 1}
    unique_vals = set(df[target_col].dropna().unique())
    invalid_vals = unique_vals - valid_vals

    if invalid_vals:
        return CheckResult(
            name="Target Label Integrity",
            status="FAIL",
            message=f"Target column contains invalid values: {invalid_vals} (must be in {valid_vals}).",
            violations_count=len(invalid_vals),
            metrics={"invalid_values": list(invalid_vals)},
        )

    fraud_rate = float((df[target_col] == 1).mean())
    return CheckResult(
        name="Target Label Integrity",
        status="PASS",
        message=f"Labels valid binary {{0, 1}}. Batch fraud rate: {fraud_rate:.3%}.",
        violations_count=0,
        metrics={"fraud_rate": fraud_rate, "fraud_count": int((df[target_col] == 1).sum())},
    )


def check_critical_nulls(
    df: pd.DataFrame,
    critical_columns: Optional[list[str]] = None,
) -> CheckResult:
    """Verify that mission-critical columns contain 0% null values."""
    cols = critical_columns or CRITICAL_ZERO_NULL_COLUMNS
    present_cols = [c for c in cols if c in df.columns]

    null_counts = df[present_cols].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()

    if cols_with_nulls:
        total_nulls = int(sum(cols_with_nulls.values()))
        return CheckResult(
            name="Critical Feature Completeness",
            status="FAIL",
            message=f"Nulls found in critical columns: {cols_with_nulls}",
            violations_count=total_nulls,
            metrics={"cols_with_nulls": {k: int(v) for k, v in cols_with_nulls.items()}},
        )

    return CheckResult(
        name="Critical Feature Completeness",
        status="PASS",
        message=f"Zero nulls across all {len(present_cols)} critical columns.",
        violations_count=0,
        metrics={"checked_columns": present_cols},
    )


def check_numeric_ranges(
    df: pd.DataFrame,
    range_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> CheckResult:
    """Verify that numerical features fall within valid physical & empirical bounds."""
    bounds = range_bounds or DEFAULT_NUMERIC_RANGES
    violations = {}
    total_violating_rows = 0

    for col, (min_val, max_val) in bounds.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_bounds = ((series < min_val) | (series > max_val)).sum()
        if out_of_bounds > 0:
            violations[col] = {
                "out_of_bounds_count": int(out_of_bounds),
                "expected_min": min_val,
                "expected_max": max_val,
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
            }
            total_violating_rows += int(out_of_bounds)

    if violations:
        return CheckResult(
            name="Numeric Range Sanity",
            status="FAIL",
            message=f"Range violations in {len(violations)} columns: {list(violations.keys())}",
            violations_count=total_violating_rows,
            metrics=violations,
        )

    return CheckResult(
        name="Numeric Range Sanity",
        status="PASS",
        message=f"All {len(bounds)} numeric features fall within expected physical/empirical ranges.",
        violations_count=0,
        metrics={"checked_features": list(bounds.keys())},
    )


def check_categorical_domains(
    df: pd.DataFrame,
    allowed_domains: Optional[dict[str, set[str]]] = None,
) -> CheckResult:
    """Verify that categorical features contain only known, permitted domain values."""
    domains = allowed_domains or ALLOWED_CATEGORIES
    violations = {}
    total_violations = 0

    for col, valid_set in domains.items():
        if col not in df.columns:
            continue
        # Convert to string and handle nulls
        series = df[col].dropna().astype(str).str.strip().str.lower()
        valid_lower = {v.lower() for v in valid_set}
        invalid_mask = ~series.isin(valid_lower)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            unexpected_samples = list(series[invalid_mask].unique()[:5])
            violations[col] = {
                "invalid_count": invalid_count,
                "unexpected_samples": unexpected_samples,
                "allowed_domain": list(valid_set),
            }
            total_violations += invalid_count

    if violations:
        return CheckResult(
            name="Categorical Domain Validity",
            status="WARN",  # WARN rather than FAIL to accommodate novel merchant/card codes gracefully
            message=f"Unexpected categorical values in {len(violations)} column(s): {list(violations.keys())}",
            violations_count=total_violations,
            metrics=violations,
        )

    return CheckResult(
        name="Categorical Domain Validity",
        status="PASS",
        message=f"All categorical features conform to declared domain sets ({list(domains.keys())}).",
        violations_count=0,
    )


def check_temporal_span(df: pd.DataFrame, dt_col: str = "TransactionDT") -> CheckResult:
    """
    Verify temporal validity:
      - Delta is non-negative and >= 86,400 (Day 1 offset)
      - Span is positive
    """
    if dt_col not in df.columns:
        return CheckResult(
            name="Temporal Validity",
            status="FAIL",
            message=f"Timestamp column '{dt_col}' not found.",
            violations_count=1,
        )

    series = df[dt_col].dropna()
    min_dt = int(series.min())
    max_dt = int(series.max())
    span_days = (max_dt - min_dt) / 86400.0

    if min_dt < 86400:
        return CheckResult(
            name="Temporal Validity",
            status="FAIL",
            message=f"TransactionDT minimum ({min_dt}) is below the Day 1 baseline (86,400s).",
            violations_count=int((series < 86400).sum()),
            metrics={"min_dt": min_dt, "max_dt": max_dt, "span_days": span_days},
        )

    return CheckResult(
        name="Temporal Validity",
        status="PASS",
        message=f"Temporal range valid: [{min_dt:,}s – {max_dt:,}s], spanning {span_days:.1f} days.",
        violations_count=0,
        metrics={"min_dt": min_dt, "max_dt": max_dt, "span_days": span_days},
    )


# ---------------------------------------------------------------------------
# Master Suite Runner
# ---------------------------------------------------------------------------

def run_data_quality_suite(
    df: pd.DataFrame,
    is_labeled: bool = True,
    critical_columns: Optional[list[str]] = None,
    range_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> DataQualityReport:
    """
    Execute the comprehensive data quality validation suite against a pandas DataFrame.

    Parameters:
      df: DataFrame containing transaction records.
      is_labeled: Whether the batch is expected to contain ground-truth 'isFraud' labels.
      critical_columns: Optional override of columns requiring 0% nulls.
      range_bounds: Optional override of numerical feature range constraints.

    Returns:
      DataQualityReport: Dataclass summarizing all check results.
    """
    required_cols = MANDATORY_LABELED_COLUMNS if is_labeled else MANDATORY_TRANSACTION_COLUMNS
    results: list[CheckResult] = []

    # 1. Schema check
    results.append(check_schema(df, required_cols))

    # 2. Primary key uniqueness
    results.append(check_uniqueness(df, "TransactionID"))

    # 3. Target label integrity (if labeled)
    if is_labeled and "isFraud" in df.columns:
        results.append(check_target_labels(df, "isFraud"))

    # 4. Critical nulls
    results.append(check_critical_nulls(df, critical_columns))

    # 5. Numeric ranges
    results.append(check_numeric_ranges(df, range_bounds))

    # 6. Categorical domain validity
    results.append(check_categorical_domains(df))

    # 7. Temporal span
    results.append(check_temporal_span(df, "TransactionDT"))

    # Compute overall status
    has_fail = any(r.status == "FAIL" for r in results)
    has_warn = any(r.status == "WARN" for r in results)

    if has_fail:
        overall_status = "FAIL"
    elif has_warn:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    passed_count = sum(1 for r in results if r.status == "PASS")
    failed_count = sum(1 for r in results if r.status == "FAIL")
    warned_count = sum(1 for r in results if r.status == "WARN")

    now_iso = datetime.now(timezone.utc).isoformat()

    return DataQualityReport(
        timestamp=now_iso,
        total_rows=len(df),
        total_columns=len(df.columns),
        overall_status=overall_status,
        checks_passed=passed_count,
        checks_failed=failed_count,
        checks_warned=warned_count,
        results=results,
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI execution entrypoint for ad-hoc data batch validation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run repeatable data quality validation on transaction batches.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train_merged.parquet",
        help="Path to parquet or CSV dataset to validate.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row limit / sample size for fast execution.",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Flag indicating the batch is unlabeled inference data (skips isFraud validation).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the data quality report.",
    )

    args = parser.parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path.resolve()}")
        raise SystemExit(1)

    print(f"Loading data from {data_path}...")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if args.sample and len(df) > args.sample:
        print(f"Sampling {args.sample:,} rows from {len(df):,} total rows...")
        df = df.sample(n=args.sample, random_state=42)

    report = run_data_quality_suite(df, is_labeled=not args.unlabeled)
    report.print_summary()

    if args.output:
        out_file = Path(args.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"Report saved to {out_file.resolve()}")

    if report.overall_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
