"""
tests/test_business_decision.py
--------------------------------
Automated test suite for Week 8 Business Decision Workflow, 3-tier economic cost model,
capacity-constrained policy solver, and sensitivity matrix calculations.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.threshold_analysis import (
    calculate_3tier_cost,
    compute_36_scenario_sensitivity_matrix,
    compute_stepup_sensitivity_matrix,
    find_capacity_constrained_policy,
    sweep_thresholds,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = _PROJECT_ROOT / "data" / "processed" / "business_decision_summary.json"


@pytest.fixture(scope="module")
def synthetic_eval_data():
    """Generate synthetic test scores and labels for fast unit testing."""
    np.random.seed(42)
    n = 10000
    y_true = (np.random.rand(n) < 0.035).astype(int)  # 3.5% fraud rate
    # Simulated model scores: frauds skewed higher, legits skewed lower
    y_prob = np.where(y_true == 1, np.random.beta(3, 2, n), np.random.beta(1, 8, n))
    return y_true, y_prob


# -----------------------------------------------------------------------------
# 1. 3-Tier Cost Model Tests
# -----------------------------------------------------------------------------
def test_3tier_cost_calculation_validity(synthetic_eval_data):
    """Verify 3-tier routing economic cost calculation consistency and non-negativity."""
    y_true, y_prob = synthetic_eval_data
    res = calculate_3tier_cost(
        y_true=y_true,
        y_prob=y_prob,
        tau_med=0.05,
        tau_high=0.60,
        fraud_loss=200.0,
        review_cost=8.0,
        stepup_cost=0.50,
        stepup_efficiency=0.80,
    )

    assert res["total_expected_cost"] >= 0.0
    assert 0.0 <= res["recall_high_tier_pct"] <= 100.0
    assert 0.0 <= res["recall_total_system_pct"] <= 100.0
    assert res["recall_total_system_pct"] >= res["recall_high_tier_pct"]
    assert 0.0 <= res["manual_review_rate_pct"] <= 100.0
    assert 0.0 <= res["stepup_rate_pct"] <= 100.0
    assert round(res["cost_accept_all"] - res["total_expected_cost"], 2) == res["net_savings_vs_accept_all"]


# -----------------------------------------------------------------------------
# 2. Threshold Sweep & Monotonicity Trend Tests
# -----------------------------------------------------------------------------
def test_threshold_sweep_bounds_and_recall_trend(synthetic_eval_data):
    """Verify threshold sweep produces valid bounds and non-increasing recall."""
    y_true, y_prob = synthetic_eval_data
    sweep = sweep_thresholds(y_true, y_prob, fraud_loss=200.0, review_cost=8.0, n_thresholds=20)

    assert len(sweep) == 20
    recalls = [pt["recall_pct"] for pt in sweep]
    thresholds = [pt["threshold"] for pt in sweep]

    # Thresholds strictly ascending
    assert thresholds == sorted(thresholds)

    # Recall is non-increasing with threshold
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1], f"Recall increased from {recalls[i]} to {recalls[i+1]} at step {i}"


# -----------------------------------------------------------------------------
# 3. Capacity-Constrained Solver Tests
# -----------------------------------------------------------------------------
def test_capacity_constrained_solver(synthetic_eval_data):
    """Assert capacity-constrained solver respects maximum review limits."""
    y_true, y_prob = synthetic_eval_data

    for cap in [1.0, 3.0, 5.0, 10.0]:
        opt = find_capacity_constrained_policy(
            y_true=y_true,
            y_prob=y_prob,
            max_review_capacity_pct=cap,
            fraud_loss=200.0,
            review_cost=8.0,
        )
        assert opt["manual_review_rate_pct"] <= cap + 1e-3, (
            f"Review rate {opt['manual_review_rate_pct']}% exceeded capacity limit {cap}%"
        )
        assert opt["tau_med"] < opt["tau_high"]


# -----------------------------------------------------------------------------
# 4. Sensitivity Matrix Completeness Tests
# -----------------------------------------------------------------------------
def test_sensitivity_matrix_completeness(synthetic_eval_data):
    """Verify 36-scenario matrix contains all 36 combinations with valid costs."""
    y_true, y_prob = synthetic_eval_data
    matrix = compute_36_scenario_sensitivity_matrix(y_true, y_prob)

    assert len(matrix) == 36
    for sc in matrix:
        assert sc["total_expected_cost"] > 0.0
        assert sc["manual_review_rate_pct"] <= sc["capacity_cap_pct"] + 1e-3


def test_stepup_sensitivity_matrix(synthetic_eval_data):
    """Verify step-up sensitivity matrix produces all 12 combinations."""
    y_true, y_prob = synthetic_eval_data
    stepup_matrix = compute_stepup_sensitivity_matrix(
        y_true, y_prob, tau_med=0.05, tau_high=0.70, fraud_loss=200.0, review_cost=8.0
    )

    assert len(stepup_matrix) == 12  # 4 efficiencies x 3 costs
    for row in stepup_matrix:
        assert row["total_expected_cost"] > 0.0
        assert 50.0 <= row["stepup_efficiency_pct"] <= 90.0


# -----------------------------------------------------------------------------
# 5. Manifest Integrity & Deliverable Tests
# -----------------------------------------------------------------------------
def test_decision_manifest_integrity():
    """Verify business_decision_summary.json exists and contains complete analysis outputs."""
    assert MANIFEST_PATH.exists(), f"Manifest file {MANIFEST_PATH} does not exist."
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check required top-level keys
    assert "metadata" in manifest
    assert "candidate_policies" in manifest
    assert "baselines_comparison" in manifest
    assert "financial_sensitivity_matrix_36_scenarios" in manifest
    assert "stepup_authentication_sensitivity" in manifest
    assert "unseen_entity_stress_test" in manifest

    # Check Candidate Policies
    policies = manifest["candidate_policies"]
    assert "policy_a_conservative" in policies
    assert "policy_b_balanced" in policies
    assert "policy_c_aggressive" in policies

    for p_key, p_val in policies.items():
        assert p_val["net_savings_vs_accept_all"] > 0.0
        assert p_val["tau_med"] < p_val["tau_high"]

    # Check Baselines
    baselines = manifest["baselines_comparison"]
    assert "cost_accept_all" in baselines
    assert "cost_review_all" in baselines
    assert "naive_amount_heuristic" in baselines
    assert "logistic_regression_default" in baselines

    # Check Sensitivity Matrix
    assert len(manifest["financial_sensitivity_matrix_36_scenarios"]) == 36
