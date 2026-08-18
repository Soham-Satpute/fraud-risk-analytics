"""
src/explainability/__init__.py
------------------------------
Explainability & Grounded GenAI Narrative Generation Package.
"""

from src.explainability.llm_client import (
    DeterministicTemplateProvider,
    GrokProvider,
    LLMClient,
    OllamaProvider,
)
from src.explainability.narrative_generator import (
    GroundedNarrativeGenerator,
    NarrativeGenerationResult,
)
from src.explainability.reason_codes import (
    BusinessDecisionPolicy,
    ReasonCode,
    ReasonCodeEngine,
    TransactionExplanationPayload,
)
from src.explainability.shap_explainer import FraudSHAPExplainer

__all__ = [
    "BusinessDecisionPolicy",
    "ReasonCode",
    "ReasonCodeEngine",
    "TransactionExplanationPayload",
    "FraudSHAPExplainer",
    "LLMClient",
    "OllamaProvider",
    "GrokProvider",
    "DeterministicTemplateProvider",
    "GroundedNarrativeGenerator",
    "NarrativeGenerationResult",
]
