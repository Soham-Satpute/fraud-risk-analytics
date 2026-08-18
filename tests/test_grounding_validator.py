"""
tests/test_grounding_validator.py
---------------------------------
Pytest suite for Grounding Validator and Fallback-on-Rejection safeguards.

Covers:
  1. Direct numeric fact checking.
  2. Derived fact re-calculation (multiples, ratios).
  3. Directional consistency (+/- SHAP attributions).
  4. Speculation & hallucination detection.
  5. Automated fallback substitution on validation rejection.
"""

from __future__ import annotations

import pytest

from src.explainability.llm_client import LLMClient
from src.explainability.narrative_generator import GroundedNarrativeGenerator
from src.validation.grounding_validator import GroundingValidator


@pytest.fixture
def sample_payload() -> dict:
    """Fixture providing a standard structured SHAP evidence payload."""
    return {
        "transaction_id": 3421590,
        "fraud_probability": 0.8425,
        "predicted_risk_tier": "HIGH",
        "decision_action": "MANUAL_REVIEW",
        "recommended_workflow": "Route to prioritized manual fraud investigation queue.",
        "base_value_log_odds": -3.312,
        "model_version": "v1.0.0-champion-lgbm",
        "top_risk_factors": [
            {
                "feature": "amt_zscore_card1",
                "display_name": "Card Amount Deviation (Z-Score)",
                "feature_value": 4.15,
                "shap_value": 1.42,
                "contribution_pct": 52.3,
                "direction": "INCREASES_RISK",
                "description": "Standardized deviation of transaction amount from card historical mean",
            },
            {
                "feature": "email_match_flag",
                "display_name": "Payer-Recipient Email Match Flag",
                "feature_value": 1,
                "shap_value": 0.85,
                "contribution_pct": 31.4,
                "direction": "INCREASES_RISK",
                "description": "Identical purchaser and recipient domains",
            },
        ],
        "top_mitigating_factors": [
            {
                "feature": "D1",
                "display_name": "Days Since Prior Transaction (D1)",
                "feature_value": 45.0,
                "shap_value": -0.65,
                "contribution_pct": 100.0,
                "direction": "REDUCES_RISK",
                "description": "Elapsed days since the last recorded transaction",
            }
        ],
        "context_attributes": {
            "TransactionID": 3421590,
            "TransactionAmt": 450.0,
            "ProductCD": "W",
            "card1": 10000,
            "P_emaildomain": "gmail.com",
            "amt_ratio_mean_card1": 2.5,
        },
    }


class TestGroundingValidator:
    """Tests for factual grounding audit rules."""

    def test_clean_grounded_narrative_passes(self, sample_payload: dict) -> None:
        validator = GroundingValidator()

        valid_narrative = """### FRAUD RISK ASSESSMENT: HIGH (Score: 0.8425)
**Transaction ID:** 3421590
**Decision Action:** `MANUAL_REVIEW`

#### Primary Risk Drivers:
- **Card Amount Deviation (Z-Score)**: Observed value 4.15 (+1.42 SHAP log-odds). Transaction amount is $450.0.
- **Payer-Recipient Email Match Flag**: Observed value 1 (+0.85 SHAP log-odds).

#### Mitigating Factors:
- **Days Since Prior Transaction (D1)**: Observed value 45.0 (-0.65 SHAP log-odds).

#### Recommended Workflow:
Route to prioritized manual fraud investigation queue."""

        res = validator.validate_narrative(valid_narrative, sample_payload)
        assert res.is_grounded is True
        assert res.grounding_score >= 0.85
        assert len(res.rejection_reasons) == 0
        assert len(res.unsupported_numbers) == 0

    def test_derived_multiple_recalculation(self, sample_payload: dict) -> None:
        validator = GroundingValidator()

        # Narrative contains derived multiple "2.5x the card baseline" matching amt_ratio_mean_card1 = 2.5
        narrative_with_derived = """### FRAUD RISK ASSESSMENT: HIGH (Score: 0.8425)
**Transaction ID:** 3421590
**Decision Action:** `MANUAL_REVIEW`

#### Primary Risk Drivers:
- **Card Amount Deviation (Z-Score)**: Observed value 4.15 (+1.42 SHAP log-odds). Amount is approximately 2.5x the card average.

#### Mitigating Factors:
- **Days Since Prior Transaction (D1)**: Observed value 45.0 (-0.65 SHAP log-odds).

#### Recommended Workflow:
Route to prioritized manual fraud investigation queue."""

        res = validator.validate_narrative(narrative_with_derived, sample_payload)
        assert res.derived_facts_verified >= 1
        assert res.is_grounded is True

    def test_unsupported_fabricated_number_fails(self, sample_payload: dict) -> None:
        validator = GroundingValidator()

        # Narrative includes fabricated number $9999.00 and 12.8 sigma not present in payload
        hallucinated_narrative = """### FRAUD RISK ASSESSMENT: HIGH (Score: 0.8425)
**Transaction ID:** 3421590
**Decision Action:** `MANUAL_REVIEW`

#### Primary Risk Drivers:
- **Card Amount Deviation**: Amount is $9999.00 with extreme 12.8 sigma spike.

#### Mitigating Factors:
- None.

#### Recommended Workflow:
Manual review."""

        res = validator.validate_narrative(hallucinated_narrative, sample_payload)
        assert res.is_grounded is False
        assert len(res.unsupported_numbers) > 0
        assert any("9999" in un or "12.8" in un for un in res.unsupported_numbers)

    def test_directional_inversion_fails(self, sample_payload: dict) -> None:
        validator = GroundingValidator()

        # Inverting D1 (which has negative SHAP -0.65) and placing it under Primary Risk Drivers
        inverted_narrative = """### FRAUD RISK ASSESSMENT: HIGH (Score: 0.8425)
**Transaction ID:** 3421590
**Decision Action:** `MANUAL_REVIEW`

#### Primary Risk Drivers:
- **Days Since Prior Transaction (D1)**: Observed value 45.0 is the main reason this transaction is high risk.

#### Mitigating Factors:
- None.

#### Recommended Workflow:
Manual review."""

        res = validator.validate_narrative(inverted_narrative, sample_payload)
        assert res.is_grounded is False
        assert len(res.directional_violations) > 0

    def test_forbidden_speculation_fails(self, sample_payload: dict) -> None:
        validator = GroundingValidator()

        speculative_narrative = """### FRAUD RISK ASSESSMENT: HIGH (Score: 0.8425)
**Transaction ID:** 3421590
**Decision Action:** `MANUAL_REVIEW`

#### Primary Risk Drivers:
- The card was confirmed stolen in a dark web leak by a known fraud ring.

#### Mitigating Factors:
- None.

#### Recommended Workflow:
Manual review."""

        res = validator.validate_narrative(speculative_narrative, sample_payload)
        assert res.is_grounded is False
        assert len(res.speculation_violations) > 0
        assert any("dark web" in sv for sv in res.speculation_violations)


class TestGroundedNarrativeGeneratorFallback:
    """Tests for automatic fallback substitution upon validation failure."""

    class MockHallucinatingProvider:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "This transaction is fraudulent because of a dark web alert with $88,888.00 stolen."

    def test_generator_fallback_on_rejection(self, sample_payload: dict) -> None:
        client = LLMClient()
        # Inject mock provider that produces hallucinated output
        client.ollama_engine = self.MockHallucinatingProvider()  # type: ignore

        generator = GroundedNarrativeGenerator(llm_client=client, preferred_provider="ollama")
        res = generator.generate_narrative_for_payload(sample_payload)

        # Fallback substitution should have triggered
        assert res.is_fallback_substituted is True
        assert "deterministic_fallback" in res.provider_used
        # Final substituted narrative should pass validation
        assert res.grounding_validation.is_grounded is True
        assert "FRAUD RISK ASSESSMENT: HIGH" in res.narrative_text
        assert "MANUAL_REVIEW" in res.narrative_text
