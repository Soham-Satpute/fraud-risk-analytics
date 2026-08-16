"""
tests/test_feature_engineering.py
---------------------------------
Pytest test suite for Week 3 feature engineering pipeline.

Covers:
  1. Mathematical correctness of transformations (log, z-score, cyclical sin/cos).
  2. Strict temporal leakage prevention (training statistics frozen and isolated from test).
  3. Graceful handling of novel/unseen categories and zero-variance groups.
  4. Pipeline serialization/deserialization integrity.
  5. Invariant checks (no NaNs or infinities introduced in engineered outputs).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    FREQUENCY_ENCODE_COLUMNS,
    REFERENCE_DT_ORIGIN,
    FraudFeaturePipeline,
)

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Fixture providing a deterministic training dataset."""
    return pd.DataFrame({
        "TransactionID": [101, 102, 103, 104, 105, 106],
        "isFraud": [0, 0, 1, 0, 1, 0],
        "TransactionDT": [
            86400,          # Hour 0, Day 0
            86400 + 21600,  # Hour 6, Day 0
            86400 + 43200,  # Hour 12, Day 0
            86400 + 64800,  # Hour 18, Day 0
            86400 + 86400,  # Hour 0, Day 1
            86400 + 172800, # Hour 0, Day 2
        ],
        "TransactionAmt": [10.0, 20.0, 30.0, 100.0, 200.0, 50.0],
        "card1": [1000, 1000, 1000, 2000, 2000, 3000],
        "addr1": [150.0, 150.0, 150.0, 200.0, 200.0, 300.0],
        "ProductCD": ["W", "W", "C", "W", "R", "W"],
        "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "gmail.com", "yahoo.com", "hotmail.com"],
        "R_emaildomain": ["gmail.com", "yahoo.com", "yahoo.com", None, "yahoo.com", None],
        "C1": [1.0, 2.0, 3.0, 1.0, 4.0, 1.0],
        "D1": [0.0, 5.0, 10.0, 0.0, 12.0, 0.0],
    })


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Fixture providing a test dataset with unseen categories and novel values."""
    return pd.DataFrame({
        "TransactionID": [201, 202, 203],
        "isFraud": [0, 1, 0],
        "TransactionDT": [
            86400 + 300000,
            86400 + 400000,
            86400 + 500000,
        ],
        "TransactionAmt": [20.0, 500.0, 75.0],
        "card1": [1000, 9999, 2000],          # 9999 is unseen in train
        "addr1": [150.0, 999.0, 200.0],       # 999.0 is unseen in train
        "ProductCD": ["W", "UNKNOWN", "R"],    # UNKNOWN is unseen in train
        "P_emaildomain": ["gmail.com", "novel.org", "yahoo.com"], # novel.org unseen
        "R_emaildomain": ["gmail.com", None, "novel.org"],
        "C1": [2.0, 1.0, 5.0],
        "D1": [1.0, 0.0, 8.0],
    })


# ---------------------------------------------------------------------------
# Unit Tests — Mathematical & Transformation Integrity
# ---------------------------------------------------------------------------

class TestMathematicalTransformations:
    """Tests verifying transformation calculations."""

    def test_log_transformation(self, synthetic_train_df: pd.DataFrame) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        transformed = pipeline.transform(synthetic_train_df)

        expected_log = np.log1p(synthetic_train_df["TransactionAmt"].values)
        np.testing.assert_allclose(transformed["log_TransactionAmt"].values, expected_log, rtol=1e-5)

    def test_cyclical_hour_encoding(self, synthetic_train_df: pd.DataFrame) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        transformed = pipeline.transform(synthetic_train_df)

        sin_vals = transformed["hour_sin"].values
        cos_vals = transformed["hour_cos"].values

        # All values in [-1, 1]
        assert np.all(sin_vals >= -1.0) and np.all(sin_vals <= 1.0)
        assert np.all(cos_vals >= -1.0) and np.all(cos_vals <= 1.0)

        # sin^2 + cos^2 = 1.0
        np.testing.assert_allclose(sin_vals**2 + cos_vals**2, 1.0, atol=1e-4)

        # Exact quadrant checks:
        # Row 0 (Hour 0): sin=0, cos=1
        assert abs(sin_vals[0] - 0.0) < 1e-4
        assert abs(cos_vals[0] - 1.0) < 1e-4
        # Row 1 (Hour 6): sin=1, cos=0
        assert abs(sin_vals[1] - 1.0) < 1e-4
        assert abs(cos_vals[1] - 0.0) < 1e-4
        # Row 2 (Hour 12): sin=0, cos=-1
        assert abs(sin_vals[2] - 0.0) < 1e-4
        assert abs(cos_vals[2] - (-1.0)) < 1e-4
        # Row 3 (Hour 18): sin=-1, cos=0
        assert abs(sin_vals[3] - (-1.0)) < 1e-4
        assert abs(cos_vals[3] - 0.0) < 1e-4

    def test_cyclical_dow_encoding(self, synthetic_train_df: pd.DataFrame) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        transformed = pipeline.transform(synthetic_train_df)

        dow_sin = transformed["dow_sin"].values
        dow_cos = transformed["dow_cos"].values

        assert np.all(dow_sin >= -1.0) and np.all(dow_sin <= 1.0)
        assert np.all(dow_cos >= -1.0) and np.all(dow_cos <= 1.0)
        np.testing.assert_allclose(dow_sin**2 + dow_cos**2, 1.0, atol=1e-4)


class TestAmountZScoreCalculations:
    """Tests for amount z-score and deviation metrics."""

    def test_card1_zscore_exact(self, synthetic_train_df: pd.DataFrame) -> None:
        # Card 1000 has amounts [10, 20, 30] -> mean = 20, sample std = 10 (or population std approx 8.16)
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        transformed = pipeline.transform(synthetic_train_df)

        # For amount 20.0 (mean), diff should be 0 and z-score should be 0
        assert abs(transformed.loc[1, "amt_diff_mean_card1"]) < 1e-4
        assert abs(transformed.loc[1, "amt_zscore_card1"]) < 1e-4
        assert abs(transformed.loc[1, "amt_ratio_mean_card1"] - 1.0) < 1e-4

        # For amount 10.0 (< mean), diff is negative, z-score is negative
        assert transformed.loc[0, "amt_diff_mean_card1"] < 0
        assert transformed.loc[0, "amt_zscore_card1"] < 0

        # For amount 30.0 (> mean), diff is positive, z-score is positive
        assert transformed.loc[2, "amt_diff_mean_card1"] > 0
        assert transformed.loc[2, "amt_zscore_card1"] > 0


class TestTemporalLeakagePrevention:
    """Tests ensuring no test distribution signals bleed into train features."""

    def test_unseen_categories_fallback(
        self, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame
    ) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        test_transformed = pipeline.transform(synthetic_test_df)

        # Row 1 has card1=9999 (unseen in train) -> frequency should be 0.0
        assert test_transformed.loc[1, "freq_card1"] == 0.0

        # Row 1 has ProductCD='UNKNOWN' -> frequency should be 0.0
        assert test_transformed.loc[1, "freq_ProductCD"] == 0.0

        # Row 1 has P_emaildomain='novel.org' -> frequency should be 0.0
        assert test_transformed.loc[1, "freq_P_emaildomain"] == 0.0

        # Unseen card1 uses global mean fallback without NaN or inf
        assert not np.isnan(test_transformed.loc[1, "amt_zscore_card1"])
        assert not np.isinf(test_transformed.loc[1, "amt_zscore_card1"])

    def test_frozen_training_lookups_consistency(
        self, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame
    ) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        
        # Capture lookup states
        initial_card1_mean = pipeline.card1_amt_stats[1000][0]
        
        # Transform test data (which contains extreme amount 500.0)
        _ = pipeline.transform(synthetic_test_df)

        # Verify lookup tables remain strictly identical
        assert pipeline.card1_amt_stats[1000][0] == initial_card1_mean


class TestEmailFlags:
    """Tests for email matching and null indicator features."""

    def test_email_match_flag(self, synthetic_train_df: pd.DataFrame) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        transformed = pipeline.transform(synthetic_train_df)

        # Row 0: gmail.com == gmail.com -> 1
        assert transformed.loc[0, "email_match_flag"] == 1
        # Row 1: gmail.com != yahoo.com -> 0
        assert transformed.loc[1, "email_match_flag"] == 0
        # Row 2: yahoo.com == yahoo.com -> 1
        assert transformed.loc[2, "email_match_flag"] == 1
        # Row 3: R_email is None -> 0
        assert transformed.loc[3, "email_match_flag"] == 0


class TestPipelineSerialization:
    """Tests verifying pipeline save and load round-trip."""

    def test_save_and_load_pipeline(
        self, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame
    ) -> None:
        pipeline = FraudFeaturePipeline().fit(synthetic_train_df)
        orig_transformed = pipeline.transform(synthetic_test_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_pipeline.joblib"
            pipeline.save(tmp_path)

            loaded_pipeline = FraudFeaturePipeline.load(tmp_path)
            loaded_transformed = loaded_pipeline.transform(synthetic_test_df)

            # Assert identical output columns and values
            assert list(orig_transformed.columns) == list(loaded_transformed.columns)
            for col in pipeline.engineered_feature_names:
                np.testing.assert_allclose(
                    orig_transformed[col].values,
                    loaded_transformed[col].values,
                    rtol=1e-5,
                    err_msg=f"Mismatch in feature {col}",
                )
