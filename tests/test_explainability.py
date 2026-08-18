"""
tests/test_explainability.py
----------------------------
Pytest suite for Week 6 explainability, SHAP reason code extraction,
V-feature collinearity consolidation, and business action resolving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.llm_client import DeterministicTemplateProvider, LLMClient
from src.explainability.reason_codes import (
    BusinessDecisionPolicy,
    ReasonCodeEngine,
    TransactionExplanationPayload,
)
from src.explainability.shap_explainer import FraudSHAPExplainer


class TestBusinessDecisionPolicy:
    """Tests for business action and risk tier mapping."""

    def test_risk_tier_resolutions(self) -> None:
        policy = BusinessDecisionPolicy(threshold_medium=0.10, threshold_high=0.35)

        # Low risk
        tier_low, action_low, work_low = policy.resolve_tier_and_action(0.04)
        assert tier_low == "LOW"
        assert action_low == "APPROVE"
        assert "straight-through" in work_low

        # Medium risk
        tier_med, action_med, work_med = policy.resolve_tier_and_action(0.22)
        assert tier_med == "MEDIUM"
        assert action_med == "STEP_UP_AUTH"
        assert "step-up" in work_med

        # High risk
        tier_high, action_high, work_high = policy.resolve_tier_and_action(0.85)
        assert tier_high == "HIGH"
        assert action_high == "MANUAL_REVIEW"
        assert "manual" in work_high


class TestReasonCodeEngine:
    """Tests for reason code extraction, human mappings, and V-collinearity consolidation."""

    def test_collinear_v_consolidation(self) -> None:
        engine = ReasonCodeEngine()

        # Near-duplicate collinear V features from the V95/V101/V279/V293 cluster
        feature_names = ["V95", "V101", "V279", "amt_zscore_card1", "C1"]
        feature_values = [5.0, 5.0, 5.0, 3.5, 12.0]
        shap_values = np.array([1.20, 1.18, 1.15, 0.95, -0.40])

        top_risk, top_mitigating = engine.consolidate_and_extract_reason_codes(
            feature_names=feature_names,
            feature_values=feature_values,
            shap_values=shap_values,
            top_k=5,
        )

        risk_features = [rc.feature for rc in top_risk]
        assert "V95" in risk_features
        assert "V101" not in risk_features
        assert "V279" not in risk_features
        assert "amt_zscore_card1" in risk_features

        assert len(top_mitigating) == 1
        assert top_mitigating[0].feature == "C1"
        assert top_mitigating[0].direction == "REDUCES_RISK"

    def test_human_feature_display_mappings(self) -> None:
        engine = ReasonCodeEngine()

        info_amt = engine.get_feature_display_info("amt_zscore_card1")
        assert "Z-Score" in info_amt["display_name"]
        assert info_amt["category"] == "AMOUNT_ANOMALY"

        info_email = engine.get_feature_display_info("email_match_flag")
        assert "Email Match" in info_email["display_name"]

        info_v = engine.get_feature_display_info("V257")
        assert "Risk" in info_v["display_name"] or "V257" in info_v["display_name"]


class TestFraudSHAPExplainer:
    """Tests for TreeSHAP explainer loading and inference explanation."""

    @pytest.fixture
    def explainer(self) -> FraudSHAPExplainer:
        return FraudSHAPExplainer()

    def test_explainer_initialization(self, explainer: FraudSHAPExplainer) -> None:
        assert explainer.model is not None
        assert len(explainer.feature_names) > 400

    def test_single_transaction_explanation(self, explainer: FraudSHAPExplainer) -> None:
        sample_dict = {
            "TransactionID": 9999999,
            "TransactionDT": 13000000,
            "TransactionAmt": 250.0,
            "ProductCD": "W",
            "card1": 10000,
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "gmail.com",
            "C1": 3.0,
            "D1": 15.0,
        }

        payload = explainer.explain_transaction(sample_dict, top_k=5)

        assert isinstance(payload, TransactionExplanationPayload)
        assert payload.transaction_id == 9999999
        assert 0.0 <= payload.fraud_probability <= 1.0
        assert payload.predicted_risk_tier in ("LOW", "MEDIUM", "HIGH")
        assert payload.decision_action in ("APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW")
        assert isinstance(payload.top_risk_factors, list)
        assert isinstance(payload.top_mitigating_factors, list)


class TestDeterministicTemplateProvider:
    """Tests for deterministic fallback narrative generation."""

    def test_format_from_payload(self) -> None:
        payload = {
            "transaction_id": 123456,
            "fraud_probability": 0.885,
            "predicted_risk_tier": "HIGH",
            "decision_action": "MANUAL_REVIEW",
            "recommended_workflow": "Route to prioritized manual fraud investigation queue.",
            "top_risk_factors": [
                {
                    "feature": "amt_zscore_card1",
                    "display_name": "Card Amount Deviation (Z-Score)",
                    "feature_value": 4.25,
                    "shap_value": 1.45,
                    "description": "Standardized deviation of transaction amount",
                }
            ],
            "top_mitigating_factors": [],
        }

        text = DeterministicTemplateProvider.format_from_payload(payload)
        assert "FRAUD RISK ASSESSMENT: HIGH" in text
        assert "0.8850" in text
        assert "MANUAL_REVIEW" in text
        assert "Card Amount Deviation" in text
        assert "4.25" in text
