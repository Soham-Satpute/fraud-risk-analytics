"""
tests/test_data_quality.py
--------------------------
Pytest test suite for automated, repeatable data quality verification.

Covers:
  1. Synthetic corruption tests: Ensures check functions strictly catch and flag invalid states.
  2. Edge case testing: Null targets, duplicate IDs, negative transaction amounts, temporal anomalies.
  3. Live batch integration tests: Runs validation against real IEEE-CIS data partitions.
  4. Temporal split verification: Verifies the Week 1 temporal cutoff (TransactionDT = 12,192,854).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.validation.data_quality import (
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

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_batch() -> pd.DataFrame:
    """Fixture providing a synthetically valid batch of transactions."""
    return pd.DataFrame({
        "TransactionID": [3000001, 3000002, 3000003, 3000004, 3000005],
        "isFraud": [0, 0, 1, 0, 0],
        "TransactionDT": [86400, 86450, 90000, 120000, 150000],
        "TransactionAmt": [50.0, 120.50, 39.99, 1000.0, 45.0],
        "ProductCD": ["W", "H", "C", "S", "R"],
        "card1": [13926, 2755, 4663, 18132, 3938],
        "card4": ["discover", "mastercard", "visa", "mastercard", "visa"],
        "card6": ["credit", "debit", "debit", "credit", "debit"],
        "C1": [1.0, 2.0, 1.0, 5.0, 1.0],
        "D1": [14.0, 0.0, 0.0, 112.0, 0.0],
    })


# ---------------------------------------------------------------------------
# Unit Tests — Synthetic Valid & Corrupted Batches
# ---------------------------------------------------------------------------

class TestSchemaIntegrity:
    """Tests for schema and required column validation."""

    def test_schema_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_schema(valid_batch, ["TransactionID", "isFraud", "TransactionAmt"])
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_schema_missing_columns(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.drop(columns=["TransactionAmt", "card1"])
        result = check_schema(corrupted, ["TransactionID", "TransactionAmt", "card1"])
        assert result.status == "FAIL"
        assert result.violations_count == 2
        assert "TransactionAmt" in result.metrics["missing_columns"]
        assert "card1" in result.metrics["missing_columns"]


class TestPrimaryKeyUniqueness:
    """Tests for TransactionID uniqueness enforcement."""

    def test_uniqueness_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_uniqueness(valid_batch, "TransactionID")
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_uniqueness_duplicate_ids(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[1, "TransactionID"] = corrupted.loc[0, "TransactionID"]
        result = check_uniqueness(corrupted, "TransactionID")
        assert result.status == "FAIL"
        assert result.violations_count == 1

    def test_uniqueness_missing_column(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.drop(columns=["TransactionID"])
        result = check_uniqueness(corrupted, "TransactionID")
        assert result.status == "FAIL"


class TestTargetLabelIntegrity:
    """Tests for fraud label validity, null checks, and allowed values."""

    def test_target_labels_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_target_labels(valid_batch, "isFraud")
        assert result.status == "PASS"
        assert result.violations_count == 0
        assert result.metrics["fraud_rate"] == 0.20  # 1 out of 5

    def test_target_labels_invalid_values(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[2, "isFraud"] = 2  # Invalid label
        result = check_target_labels(corrupted, "isFraud")
        assert result.status == "FAIL"
        assert 2 in result.metrics["invalid_values"]

    def test_target_labels_nulls(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy().astype({"isFraud": "float64"})
        corrupted.loc[1, "isFraud"] = np.nan
        result = check_target_labels(corrupted, "isFraud")
        assert result.status == "FAIL"
        assert result.violations_count == 1


class TestCriticalNulls:
    """Tests for mission-critical columns completeness."""

    def test_critical_nulls_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_critical_nulls(valid_batch, ["TransactionID", "TransactionAmt", "card1"])
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_critical_nulls_present(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "TransactionAmt"] = np.nan
        corrupted.loc[3, "card1"] = np.nan
        result = check_critical_nulls(corrupted, ["TransactionAmt", "card1"])
        assert result.status == "FAIL"
        assert result.violations_count == 2


class TestNumericRanges:
    """Tests for physical and domain range boundaries."""

    def test_numeric_ranges_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_numeric_ranges(valid_batch)
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_numeric_ranges_negative_amount(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "TransactionAmt"] = -10.0
        result = check_numeric_ranges(corrupted)
        assert result.status == "FAIL"
        assert "TransactionAmt" in result.metrics

    def test_numeric_ranges_zero_amount(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "TransactionAmt"] = 0.0
        result = check_numeric_ranges(corrupted)
        assert result.status == "FAIL"

    def test_numeric_ranges_extreme_outlier(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "TransactionAmt"] = 100000.0  # Over 35k limit
        result = check_numeric_ranges(corrupted)
        assert result.status == "FAIL"


class TestCategoricalDomains:
    """Tests for categorical domain matching."""

    def test_categorical_domains_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_categorical_domains(valid_batch)
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_categorical_domains_unexpected_product(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "ProductCD"] = "UNKNOWN_PRODUCT"
        result = check_categorical_domains(corrupted)
        assert result.status == "WARN"
        assert result.violations_count == 1


class TestTemporalValidity:
    """Tests for time delta bounds and monotonicity."""

    def test_temporal_validity_valid(self, valid_batch: pd.DataFrame) -> None:
        result = check_temporal_span(valid_batch, "TransactionDT")
        assert result.status == "PASS"
        assert result.violations_count == 0

    def test_temporal_validity_before_origin(self, valid_batch: pd.DataFrame) -> None:
        corrupted = valid_batch.copy()
        corrupted.loc[0, "TransactionDT"] = 100  # Less than 86,400s (Day 1)
        result = check_temporal_span(corrupted, "TransactionDT")
        assert result.status == "FAIL"


@pytest.fixture(scope="module")
def real_sample_df() -> pd.DataFrame:
    parquet_path = Path("data/processed/train_merged.parquet")
    if not parquet_path.exists():
        pytest.skip(f"Processed parquet dataset not found at {parquet_path}")
    
    # Read 10,000 row sample for fast deterministic testing
    df = pd.read_parquet(parquet_path)
    return df.sample(n=min(10000, len(df)), random_state=42)


# ---------------------------------------------------------------------------
# Integration Tests — Real Merged Parquet Sample & Split Partition Checks
# ---------------------------------------------------------------------------

class TestLiveDatasetIntegrity:
    """Integration checks running on the actual parquet dataset if available."""

    def test_live_data_quality_suite(self, real_sample_df: pd.DataFrame) -> None:
        """Run entire suite against sample from real merged parquet."""
        report = run_data_quality_suite(real_sample_df, is_labeled=True)
        assert report.overall_status in {"PASS", "WARN"}
        assert report.checks_failed == 0

    def test_temporal_split_boundary_properties(self) -> None:
        """Verify Week 1 temporal split cutoff properties."""
        parquet_path = Path("data/processed/train_merged.parquet")
        if not parquet_path.exists():
            pytest.skip(f"Processed parquet dataset not found at {parquet_path}")

        df = pd.read_parquet(parquet_path, columns=["TransactionDT", "isFraud"])
        cutoff_dt = 12192854  # 80th percentile confirmed in Week 1

        train_part = df[df["TransactionDT"] <= cutoff_dt]
        test_part = df[df["TransactionDT"] > cutoff_dt]

        assert len(train_part) == 472432, f"Expected 472,432 train rows, got {len(train_part):,}"
        assert len(test_part) == 118108, f"Expected 118,108 test rows, got {len(test_part):,}"
        assert train_part["TransactionDT"].max() <= cutoff_dt
        assert test_part["TransactionDT"].min() > cutoff_dt
