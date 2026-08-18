"""
src/explainability/llm_client.py
--------------------------------
Multi-Provider LLM Client Interface for Offline Grounded Narrative Generation.

Hierarchy:
  1. Primary: Local Open-Source LLM via Ollama (e.g., Llama 3.2 3B, Phi-3 Mini) - $0 stack.
  2. Fallback / Baseline Guarantee: Deterministic Rule-Based Template Engine (100% reproducible).
  3. Optional Experiment: Grok / xAI API (https://api.x.ai/v1) if API key is provided.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)

# Default endpoints and models
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

DEFAULT_GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-mini")


class LLMProvider(Protocol):
    """Protocol defining the interface for narrative generation providers."""
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class OllamaProvider:
    """Primary local open-source LLM provider via Ollama."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout_seconds

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call local Ollama chat API at temperature=0."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0.0,
            },
            "stream": False,
        }
        try:
            resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", "")).strip()
        except requests.exceptions.RequestException as e:
            logger.warning("Ollama connection failed (%s).", e)
            raise RuntimeError(f"Ollama provider failed: {e}") from e


class GrokProvider:
    """Optional cloud provider calling xAI / Grok OpenAI-compatible REST endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = DEFAULT_GROK_API_URL,
        model: str = DEFAULT_GROK_MODEL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        self.api_url = api_url
        self.model = model
        self.timeout = timeout_seconds

        if not self.api_key:
            logger.debug("Grok API key not provided; GrokProvider disabled.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call xAI Grok API completions at temperature=0."""
        if not self.api_key:
            raise ValueError("GROK_API_KEY or XAI_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
            return ""
        except requests.exceptions.RequestException as e:
            logger.warning("Grok API request failed (%s).", e)
            raise RuntimeError(f"Grok provider failed: {e}") from e


class DeterministicTemplateProvider:
    """
    Baseline guarantee provider that generates strictly grounded, structured narratives
    directly from SHAP reason codes and business policy without external API or GPU dependencies.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Parse structured payload from prompt JSON and generate clean, standardized narrative.
        """
        try:
            # Extract JSON block from user prompt
            start_idx = user_prompt.find("{")
            end_idx = user_prompt.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = user_prompt[start_idx : end_idx + 1]
                data = json.loads(json_str)
                return self.format_from_payload(data)
        except Exception as e:
            logger.debug("Could not parse JSON from prompt (%s); falling back to direct format.", e)

        return "Deterministic baseline narrative generated from structured SHAP attributions."

    @staticmethod
    def format_from_payload(payload: dict[str, Any]) -> str:
        """Format a clean, structured analyst narrative from explanation payload."""
        tx_id = payload.get("transaction_id", "N/A")
        prob = payload.get("fraud_probability", 0.0)
        risk_tier = payload.get("predicted_risk_tier", "LOW")
        action = payload.get("decision_action", "APPROVE")
        workflow = payload.get("recommended_workflow", "Approve transaction.")
        risk_factors = payload.get("top_risk_factors", [])
        mitigating = payload.get("top_mitigating_factors", [])

        lines = [
            f"### FRAUD RISK ASSESSMENT: {risk_tier} (Score: {prob:.4f})",
            f"**Transaction ID:** {tx_id}",
            f"**Decision Action:** `{action}`",
            "",
            "#### Primary Risk Drivers:",
        ]

        if risk_factors:
            for factor in risk_factors:
                d_name = factor.get("display_name", factor.get("feature", "Unknown"))
                val = factor.get("feature_value", "N/A")
                shap = factor.get("shap_value", 0.0)
                desc = factor.get("description", "")
                lines.append(f"- **{d_name}**: Observed value `{val}` (+{shap:.3f} SHAP log-odds). {desc}")
        else:
            lines.append("- No significant elevated risk drivers observed.")

        lines.extend(["", "#### Mitigating Factors:"])
        if mitigating:
            for factor in mitigating:
                d_name = factor.get("display_name", factor.get("feature", "Unknown"))
                val = factor.get("feature_value", "N/A")
                shap = factor.get("shap_value", 0.0)
                desc = factor.get("description", "")
                lines.append(f"- **{d_name}**: Observed value `{val}` ({shap:.3f} SHAP log-odds). {desc}")
        else:
            lines.append("- No strong mitigating factors identified.")

        lines.extend([
            "",
            "#### Recommended Workflow:",
            f"{workflow}",
        ])

        return "\n".join(lines)


class LLMClient:
    """
    Unified client coordinating tiered providers with automatic fallback.
    """

    def __init__(
        self,
        preferred_provider: str = "ollama",
        fallback_provider: str = "deterministic",
        grok_api_key: str | None = None,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        grok_model: str = DEFAULT_GROK_MODEL,
    ) -> None:
        self.preferred_provider = preferred_provider.lower()
        self.fallback_provider = fallback_provider.lower()

        self.deterministic_engine = DeterministicTemplateProvider()
        self.ollama_engine = OllamaProvider(model=ollama_model)
        self.grok_engine = GrokProvider(api_key=grok_api_key, model=grok_model)

    def generate_narrative(
        self,
        system_prompt: str,
        user_prompt: str,
        force_provider: str | None = None,
    ) -> tuple[str, str]:
        """
        Generate narrative using preferred provider, falling back automatically if needed.

        Returns:
            tuple: (generated_text, provider_used)
        """
        provider = (force_provider or self.preferred_provider).lower()

        # 1. Try Grok if specified
        if provider in ("grok", "xai"):
            try:
                text = self.grok_engine.generate(system_prompt, user_prompt)
                if text:
                    return text, "grok"
            except Exception as e:
                logger.warning("Grok generation failed (%s); falling back...", e)

        # 2. Try Ollama if specified (or fallback from Grok)
        if provider == "ollama" or (provider in ("grok", "xai") and self.fallback_provider == "ollama"):
            try:
                text = self.ollama_engine.generate(system_prompt, user_prompt)
                if text:
                    return text, "ollama"
            except Exception as e:
                logger.warning("Ollama generation failed (%s); falling back to deterministic template...", e)

        # 3. Baseline Guarantee: Deterministic Template
        text = self.deterministic_engine.generate(system_prompt, user_prompt)
        return text, "deterministic_fallback"
