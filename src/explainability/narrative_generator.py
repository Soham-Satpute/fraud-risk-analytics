"""
src/explainability/narrative_generator.py
-----------------------------------------
Grounded GenAI Analyst Narrative Generator with Fallback-on-Rejection Safeguards.

Translates structured TreeSHAP reason codes and predefined business policies into
concise, professional analyst summaries while enforcing strict factual grounding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from src.explainability.llm_client import LLMClient
from src.explainability.reason_codes import TransactionExplanationPayload
from src.validation.grounding_validator import GroundingValidationResult, GroundingValidator

logger = logging.getLogger(__name__)

# Structured System Prompt enforcing factual grounding and business policy alignment
GROUNDED_SYSTEM_PROMPT = """You are a Senior Quantitative Fraud Risk Analyst at a financial institution.
Your task is to write a concise, professional risk assessment for a transaction based STRICTLY on the provided SHAP feature attributions and predefined business action.

STRICT GROUNDING RULES:
1. NEVER speculate or invent facts not present in the input JSON (e.g. do NOT mention dark web, police reports, stolen card databases, or unmentioned countries).
2. Use ONLY the exact feature names, observed values, z-scores, amounts, and SHAP directions provided in the evidence.
3. The Recommended Workflow MUST state the provided 'decision_action' and 'recommended_workflow'—do NOT invent a different operational action.
4. If describing derived multiples (e.g., '2x average'), ensure it matches the calculated ratios from the evidence.
5. Format the output clearly using these exact markdown headers:
   ### FRAUD RISK ASSESSMENT: <TIER> (Score: <PROB>)
   **Transaction ID:** <ID>
   **Decision Action:** `<ACTION>`

   #### Primary Risk Drivers:
   - <List top risk-increasing factors with observed values and impact>

   #### Mitigating Factors:
   - <List top risk-reducing factors with observed values and impact>

   #### Recommended Workflow:
   <Predefined recommended workflow stating the operational rationale>
"""


@dataclass
class NarrativeGenerationResult:
    """Output container for generated and audited narrative."""
    transaction_id: int | None
    narrative_text: str
    provider_used: str
    is_fallback_substituted: bool
    grounding_validation: GroundingValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "narrative_text": self.narrative_text,
            "provider_used": self.provider_used,
            "is_fallback_substituted": self.is_fallback_substituted,
            "grounding_validation": self.grounding_validation.to_dict(),
        }


class GroundedNarrativeGenerator:
    """
    Coordinates narrative generation, prompt construction, and grounding validation.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        validator: GroundingValidator | None = None,
        preferred_provider: str = "ollama",
        fallback_provider: str = "deterministic",
        grok_api_key: str | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient(
            preferred_provider=preferred_provider,
            fallback_provider=fallback_provider,
            grok_api_key=grok_api_key,
        )
        self.validator = validator or GroundingValidator()

    def _build_user_prompt(self, payload: dict[str, Any]) -> str:
        """Construct user prompt embedding structured evidence payload."""
        payload_json = json.dumps(payload, indent=2)
        return (
            "Analyze the following transaction evidence payload and produce a structured, grounded fraud assessment:\n\n"
            f"```json\n{payload_json}\n```\n\n"
            "Produce the concise markdown assessment strictly following the system rules."
        )

    def generate_narrative_for_payload(
        self,
        payload: TransactionExplanationPayload | dict[str, Any],
        force_provider: str | None = None,
    ) -> NarrativeGenerationResult:
        """
        Generate and validate narrative for a transaction explanation payload.

        If the LLM output fails grounding validation, it is rejected and replaced
        by the verified deterministic baseline explanation.
        """
        payload_dict = payload.to_dict() if isinstance(payload, TransactionExplanationPayload) else payload
        tx_id = payload_dict.get("transaction_id")

        user_prompt = self._build_user_prompt(payload_dict)

        # 1. Generate text via LLM client (Ollama / Grok / Deterministic)
        raw_text, provider_used = self.llm_client.generate_narrative(
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            force_provider=force_provider,
        )

        # 2. Audit generated text with Grounding Validator
        validation = self.validator.validate_narrative(raw_text, payload_dict)

        # 3. Fallback-on-Rejection Safeguard
        fallback_substituted = False
        final_text = raw_text

        if not validation.is_grounded:
            logger.warning(
                "Transaction %s: Narrative failed grounding validation (Score: %.2f, Rejections: %s). Substituting deterministic fallback.",
                tx_id,
                validation.grounding_score,
                validation.rejection_reasons,
            )
            # Generate deterministic fallback
            final_text = self.llm_client.deterministic_engine.format_from_payload(payload_dict)
            fallback_substituted = True
            provider_used = f"{provider_used}_rejected->deterministic_fallback"

            # Re-validate fallback
            validation = self.validator.validate_narrative(final_text, payload_dict)

        return NarrativeGenerationResult(
            transaction_id=tx_id,
            narrative_text=final_text,
            provider_used=provider_used,
            is_fallback_substituted=fallback_substituted,
            grounding_validation=validation,
        )
