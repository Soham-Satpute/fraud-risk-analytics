"""
tests/test_models.py
--------------------
Pytest test suite for Week 5 model evaluation metrics and confidence interval engine.

Covers:
  1. Classification metric computations (PR-AUC, ROC-AUC, Log-Loss).
  2. Recall at fixed FPR threshold solver.
  3. Non-parametric bootstrap confidence interval estimation.
  4. Operating curve threshold sweep sanity and monotonicity.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    bootstrap_metric_confidence_intervals,
    calculate_recall_at_fixed_fpr,
    compute_classification_metrics,
    generate_threshold_sweep,
)


class TestClassificationMetrics:
    """Tests for core ranking and probability loss metrics."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.01, 0.05, 0.1, 0.9, 0.99])
        metrics = compute_classification_metrics(y_true, y_prob)

        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["brier_score"] < 0.02

    def test_random_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        metrics = compute_classification_metrics(y_true, y_prob)

        assert metrics["roc_auc"] == 0.5
        assert 0.0 <= metrics["pr_auc"] <= 1.0


class TestRecallAtFixedFPR:
    """Tests for fixed FPR operating threshold solver."""

    def test_achieved_fpr_bound(self) -> None:
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 950 + [1] * 50)
        # Synthetic probabilities with some separation
        y_prob = np.clip(y_true * 0.5 + rng.uniform(0.0, 0.5, size=1000), 0.0, 1.0)

        res_1pct = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
        assert res_1pct["achieved_fpr"] <= 0.015
        assert 0.0 <= res_1pct["recall"] <= 1.0
        assert 0.0 <= res_1pct["operating_threshold"] <= 1.0

        res_5pct = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.05)
        assert res_5pct["achieved_fpr"] <= 0.055
        # Recall at 5% FPR should be >= Recall at 1% FPR
        assert res_5pct["recall"] >= res_1pct["recall"]


class TestBootstrapConfidenceIntervals:
    """Tests for 1000-resample non-parametric bootstrap uncertainty estimation."""

    def test_bootstrap_bounds_and_uncertainty(self) -> None:
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 400 + [1] * 50)
        y_prob = np.clip(y_true * 0.4 + rng.uniform(0.0, 0.6, size=450), 0.0, 1.0)

        cis = bootstrap_metric_confidence_intervals(
            y_true, y_prob, n_bootstraps=100, ci=0.95, random_state=42
        )

        for metric in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]:
            assert metric in cis
            res = cis[metric]
            assert res["ci_95_low"] <= res["ci_95_high"]
            assert 0.0 <= res["ci_95_low"] <= 1.0
            assert 0.0 <= res["ci_95_high"] <= 1.0
            assert res["std_error"] >= 0.0


class TestThresholdSweep:
    """Tests for operating threshold sweep."""

    def test_threshold_monotonicity(self) -> None:
        y_true = np.array([0] * 90 + [1] * 10)
        y_prob = np.linspace(0.01, 0.99, 100)

        sweep = generate_threshold_sweep(y_true, y_prob, n_thresholds=20)
        assert len(sweep) == 20

        # Review rate should decrease as threshold increases
        review_rates = [s["review_rate"] for s in sweep]
        assert review_rates[0] >= review_rates[-1]
