"""
tests/test_eda_insights.py
--------------------------
Pytest test suite for Week 4 statistical insights and storytelling engine.

Covers:
  1. Wilson score confidence interval mathematical bounds and edge cases.
  2. Risk ratio and relative risk calculations.
  3. Integration tests on training dataset partition to verify all 5 stories compute without error.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eda.insights import (
    calculate_risk_ratio,
    compute_all_eda_insights,
    wilson_confidence_interval,
)


class TestWilsonConfidenceInterval:
    """Tests for binomial Wilson score interval calculation."""

    def test_standard_case(self) -> None:
        low, high = wilson_confidence_interval(k=50, n=100, confidence=0.95)
        assert 0.0 <= low <= 0.50
        assert 0.50 <= high <= 1.0
        # Expected Wilson bounds for 50/100 at 95% are approx [0.4038, 0.5962]
        assert abs(low - 0.4038) < 0.01
        assert abs(high - 0.5962) < 0.01

    def test_zero_successes(self) -> None:
        low, high = wilson_confidence_interval(k=0, n=100, confidence=0.95)
        assert low == 0.0
        assert high > 0.0
        assert high < 0.05  # Standard rule of 3/n approx 0.03

    def test_all_successes(self) -> None:
        low, high = wilson_confidence_interval(k=100, n=100, confidence=0.95)
        assert low > 0.95
        assert high == 1.0

    def test_small_sample_bounds(self) -> None:
        low, high = wilson_confidence_interval(k=1, n=5, confidence=0.95)
        assert 0.0 <= low <= high <= 1.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_confidence_interval(k=105, n=100)


class TestRiskRatio:
    """Tests for risk ratio calculation."""

    def test_two_fold_risk(self) -> None:
        # Group A: 10/100 = 10%, Group B: 5/100 = 5% -> RR = 2.0
        res = calculate_risk_ratio(k_exposed=10, n_exposed=100, k_unexposed=5, n_unexposed=100)
        assert res["risk_ratio"] == 2.0
        assert res["ci_low"] < 2.0 < res["ci_high"]


class TestInsightsIntegration:
    """Integration tests on training partition."""

    @pytest.fixture(scope="module")
    def train_sample_df(self) -> pd.DataFrame:
        parquet_path = Path("data/processed/train_features.parquet")
        if not parquet_path.exists():
            pytest.skip("train_features.parquet not generated yet.")
        return pd.read_parquet(parquet_path)

    def test_compute_all_insights_structure(self, train_sample_df: pd.DataFrame) -> None:
        parquet_path = Path("data/processed/train_features.parquet")
        insights = compute_all_eda_insights(parquet_path)

        assert "story_1_diurnal_attack_window" in insights
        assert "story_2_email_topology" in insights
        assert "story_3_amount_zscores" in insights
        assert "story_4_product_channels" in insights
        assert "story_5_identity_paradox" in insights

        # Verify Story 1
        s1 = insights["story_1_diurnal_attack_window"]
        assert len(s1["hourly_breakdown"]) == 24
        assert "stakeholder_takeaway" in s1

        # Verify Story 2
        s2 = insights["story_2_email_topology"]
        assert s2["self_transfer_stats (P == R)"]["fraud_rate_pct"] > s2["cross_transfer_stats (P != R)"]["fraud_rate_pct"]
        assert s2["self_vs_cross_transfer_risk_ratio"]["risk_ratio"] > 2.5

        # Verify Story 3
        s3 = insights["story_3_amount_zscores"]
        assert len(s3["zscore_tiers"]) == 3
        assert s3["extreme_vs_normal_risk_ratio"]["risk_ratio"] > 1.2

        # Verify Story 4
        s4 = insights["story_4_product_channels"]
        assert len(s4["product_cd_breakdown"]) > 0

        # Verify Story 5
        s5 = insights["story_5_identity_paradox"]
        assert s5["identity_joined_stats"]["fraud_rate_pct"] > s5["no_identity_stats"]["fraud_rate_pct"]
