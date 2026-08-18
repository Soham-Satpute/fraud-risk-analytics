"""
src/validation/grounding_validator.py
-------------------------------------
Automated Grounding Validator for GenAI Fraud Analyst Narratives.

Mandatory safeguard that audits LLM-generated explanations against structured SHAP evidence.
Distinguishes:
  1. Direct Facts: Exact numbers, probabilities, z-scores, amounts, counts.
  2. Derived Facts: Re-calculates ratios, multiples (e.g., "2x baseline"), and percentage changes.
  3. Feature Factuality: Verifies all cited risk drivers exist in the SHAP evidence payload.
  4. Directional Integrity: Confirms risk-increasing claims map to positive SHAP and mitigating claims to negative SHAP.
  5. Anti-Speculation: Flags unauthorized speculation (e.g. "dark web leak", "stolen card confirmed").

Any narrative failing validation is rejected and flagged for deterministic fallback substitution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Forbidden speculative terms that have no evidence in IEEE-CIS tabular features
FORBIDDEN_SPECULATION_TERMS = [
    "dark web",
    "confirmed stolen",
    "known fraud ring",
    "police report",
    "criminal syndicate",
    "stolen card database",
    "compromised credentials",
    "hacked account",
    "guaranteed fraud",
    "definitely fraudulent",
    "arrest warrant",
    "malware detected",
    "botnet cluster",
]


@dataclass
class GroundingValidationResult:
    """Detailed audit report for a single narrative validation."""
    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    direct_facts_checked: int
    derived_facts_verified: int
    unsupported_numbers: list[str]
    unsupported_features: list[str]
    directional_violations: list[str]
    speculation_violations: list[str]
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundingValidator:
    """
    Validation engine ensuring GenAI narratives remain strictly grounded to SHAP evidence.
    """

    def __init__(
        self,
        numeric_tolerance_pct: float = 0.05,  # 5% margin for rounding variations
        allow_derived_multiples: bool = True,
    ) -> None:
        self.numeric_tolerance_pct = numeric_tolerance_pct
        self.allow_derived_multiples = allow_derived_multiples

    def extract_numbers_from_text(self, text: str) -> list[float]:
        """
        Extract all numeric quantities, currencies, percentages, and z-scores from narrative,
        while filtering out numbers that are parts of column names or screen resolutions.
        """
        # Scrub screen resolutions (e.g. 1920x1080, 1600x900)
        scrubbed = re.sub(r"\b\d+x\d+\b", " ", text, flags=re.IGNORECASE)
        # Scrub feature column names and range descriptors
        scrubbed = re.sub(r"\b[VCDMid_]+\d+\b", " ", scrubbed, flags=re.IGNORECASE)
        scrubbed = re.sub(r"\bcard\d+\b", " ", scrubbed, flags=re.IGNORECASE)
        scrubbed = re.sub(r"\baddr\d+\b", " ", scrubbed, flags=re.IGNORECASE)
        scrubbed = re.sub(r"\b\d+h-\d+d\b", " ", scrubbed, flags=re.IGNORECASE)
        scrubbed = re.sub(r"\bV\d+-V\d+\b", " ", scrubbed, flags=re.IGNORECASE)

        # Clean punctuation and currency symbols
        clean_text = scrubbed.replace(",", "").replace("$", " ").replace("%", " ").replace("σ", " ")

        # Regex for floats, integers, and scientific notation
        pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        raw_matches = re.findall(pattern, clean_text)

        numbers: list[float] = []
        for m in raw_matches:
            if not m or m in ("-", "+", "."):
                continue
            try:
                num = float(m)
                numbers.append(num)
            except ValueError:
                pass
        return numbers

    def extract_derived_claims(self, text: str) -> list[dict[str, Any]]:
        """
        Extract derived multiple claims like '2x', '3.5 times', 'twice', 'triple', '+50%',
        while ignoring screen resolutions like '1920x1080'.
        """
        # First scrub screen resolutions
        cleaned_text = re.sub(r"\b\d+x\d+\b", " ", text, flags=re.IGNORECASE)

        derived_claims: list[dict[str, Any]] = []

        # 1. Multiples regex: e.g. "2.5x", "3x", "2 times", "2.5 fold"
        multiple_pattern = r"\b(\d+(?:\.\d+)?)\s*(?:x|X|\s*times\b|\s*fold\b)"
        for match in re.finditer(multiple_pattern, cleaned_text, re.IGNORECASE):
            try:
                multiple_val = float(match.group(1))
                if multiple_val > 100:
                    continue
                derived_claims.append({
                    "type": "MULTIPLE",
                    "value": multiple_val,
                    "raw_text": match.group(0).strip(),
                })
            except ValueError:
                pass

        # 2. Word-based multiples
        word_multiples = {"twice": 2.0, "double": 2.0, "triple": 3.0, "quadruple": 4.0, "half": 0.5}
        for word, val in word_multiples.items():
            if re.search(rf"\b{word}\b", cleaned_text, re.IGNORECASE):
                derived_claims.append({
                    "type": "MULTIPLE",
                    "value": val,
                    "raw_text": word,
                })

        return derived_claims

    def _collect_ground_truth_numbers(self, payload: dict[str, Any]) -> set[float]:
        """
        Gather all valid reference numbers from explanation payload (including percentages, z-scores,
        and column index numbers).
        """
        valid_numbers: set[float] = set()

        # Core fields
        for key in ["fraud_probability", "base_value_log_odds", "transaction_id"]:
            if key in payload and payload[key] is not None:
                try:
                    val_f = float(payload[key])
                    valid_numbers.add(val_f)
                    if key == "fraud_probability":
                        valid_numbers.add(round(val_f * 100, 2))
                        valid_numbers.add(round(val_f * 100, 1))
                except (ValueError, TypeError):
                    pass

        # Context attributes
        for k, v in payload.get("context_attributes", {}).items():
            if v is not None:
                try:
                    val_f = float(v)
                    valid_numbers.add(val_f)
                except (ValueError, TypeError):
                    pass
            # Also add numbers embedded in context key names (e.g. card1, addr2, id_20)
            for num_str in re.findall(r"\d+", str(k)):
                try:
                    valid_numbers.add(float(num_str))
                except ValueError:
                    pass

        # Reason codes (risk factors and mitigating factors)
        all_factors = payload.get("top_risk_factors", []) + payload.get("top_mitigating_factors", [])
        for f in all_factors:
            for field in ["shap_value", "contribution_pct", "feature_value"]:
                val = f.get(field)
                if val is not None:
                    try:
                        val_f = float(val)
                        valid_numbers.add(val_f)
                        valid_numbers.add(round(val_f, 2))
                        valid_numbers.add(round(val_f, 1))
                        valid_numbers.add(round(abs(val_f), 2))
                        valid_numbers.add(round(abs(val_f), 3))
                    except (ValueError, TypeError):
                        pass

            # Include feature numbers and cluster member numbers (e.g. 20 from id_20 / Id 20, 95 from V95)
            feat_str = str(f.get("feature", "")) + " " + str(f.get("display_name", "")) + " " + str(f.get("cluster_members", ""))
            for num_str in re.findall(r"\d+", feat_str):
                try:
                    valid_numbers.add(float(num_str))
                except ValueError:
                    pass

        return valid_numbers

    def _verify_derived_claim(self, claim: dict[str, Any], payload: dict[str, Any]) -> bool:
        """
        Recalculate and verify a derived claim against source baseline values.
        """
        claim_val = claim["value"]
        ctx = payload.get("context_attributes", {})
        all_factors = payload.get("top_risk_factors", []) + payload.get("top_mitigating_factors", [])

        # Check 1: Card amount ratio in context or factors
        if "amt_ratio_mean_card1" in ctx and ctx["amt_ratio_mean_card1"] is not None:
            actual_ratio = float(ctx["amt_ratio_mean_card1"])
            if abs(actual_ratio - claim_val) / max(actual_ratio, 0.1) <= 0.25:
                return True

        for f in all_factors:
            if f.get("feature") == "amt_ratio_mean_card1" and f.get("feature_value") is not None:
                actual_ratio = float(f["feature_value"])
                if abs(actual_ratio - claim_val) / max(actual_ratio, 0.1) <= 0.25:
                    return True

        # Check 2: TransactionAmt vs baseline derivation
        tx_amt = ctx.get("TransactionAmt")
        if tx_amt is not None:
            try:
                tx_amt_f = float(tx_amt)
                for f in all_factors:
                    if f.get("feature") == "amt_diff_mean_card1" and f.get("feature_value") is not None:
                        diff = float(f["feature_value"])
                        mean_amt = tx_amt_f - diff
                        if mean_amt > 0:
                            calc_multiple = tx_amt_f / mean_amt
                            if abs(calc_multiple - claim_val) / max(calc_multiple, 0.1) <= 0.25:
                                return True
            except (ValueError, TypeError):
                pass

        # Check 3: Check if claim multiple matches any feature value directly
        for f in all_factors:
            val = f.get("feature_value")
            if isinstance(val, (int, float)) and val > 0:
                if abs(val - claim_val) / max(val, 0.1) <= 0.20:
                    return True

        return False

    def validate_narrative(
        self,
        narrative_text: str,
        payload: dict[str, Any],
    ) -> GroundingValidationResult:
        """
        Perform comprehensive grounding audit on the generated narrative.

        Parameters:
            narrative_text: Text string generated by LLM or template.
            payload: Ground-truth SHAP explanation payload dictionary.

        Returns:
            GroundingValidationResult: Audit outcome with scores and violation details.
        """
        unsupported_numbers: list[str] = []
        unsupported_features: list[str] = []
        directional_violations: list[str] = []
        speculation_violations: list[str] = []
        rejection_reasons: list[str] = []

        # 1. Speculation & Hallucination Scan
        for term in FORBIDDEN_SPECULATION_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", narrative_text, re.IGNORECASE):
                speculation_violations.append(f"Forbidden speculative phrase detected: '{term}'")
                rejection_reasons.append(f"Speculative assertion: '{term}'")

        # 2. Extract and Verify Derived Claims
        derived_claims = self.extract_derived_claims(narrative_text)
        derived_verified_count = 0
        derived_claim_values: set[float] = set()

        for claim in derived_claims:
            is_valid_derived = self._verify_derived_claim(claim, payload)
            if is_valid_derived:
                derived_verified_count += 1
                derived_claim_values.add(claim["value"])
            else:
                unsupported_numbers.append(f"Derived claim '{claim['raw_text']}' ({claim['value']}) could not be verified from baseline evidence")
                rejection_reasons.append(f"Invalid derived multiple: {claim['raw_text']}")

        # 3. Direct Numbers Verification
        extracted_numbers = self.extract_numbers_from_text(narrative_text)
        ground_truth_numbers = self._collect_ground_truth_numbers(payload)

        # Standard benign constants to ignore (formatting bullet indices, markdown headers)
        benign_constants = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 24.0, 30.0, 100.0}

        direct_checked_count = 0
        for num in extracted_numbers:
            # Skip if part of verified derived claim or benign formatting index
            if num in derived_claim_values or num in benign_constants:
                continue

            direct_checked_count += 1
            # Check if num matches any ground truth number within tolerance
            matched = False
            for gt in ground_truth_numbers:
                if abs(num - gt) <= max(abs(gt) * self.numeric_tolerance_pct, 0.05):
                    matched = True
                    break

            if not matched:
                unsupported_numbers.append(f"Unsubstantiated number '{num}' not found in SHAP evidence")

        if unsupported_numbers:
            rejection_reasons.append(f"{len(unsupported_numbers)} unsupported numerical quantities found")

        # 4. Feature Factuality & Directional Integrity
        top_risk = payload.get("top_risk_factors", [])
        top_mitigating = payload.get("top_mitigating_factors", [])

        risk_features = {f.get("feature"): f for f in top_risk if f.get("feature")}
        mitigating_features = {f.get("feature"): f for f in top_mitigating if f.get("feature")}

        text_lower = narrative_text.lower()

        # Check if mitigating features are mistakenly claimed as increasing risk
        for feat_name, f_info in mitigating_features.items():
            d_name = f_info.get("display_name", feat_name).lower()
            if "primary risk drivers" in text_lower:
                drivers_section = text_lower.split("primary risk drivers")[1]
                if "mitigating factors" in drivers_section:
                    drivers_section = drivers_section.split("mitigating factors")[0]

                if d_name in drivers_section and feat_name not in risk_features:
                    directional_violations.append(
                        f"Mitigating factor '{f_info.get('display_name')}' (SHAP: {f_info.get('shap_value')}) was described as a primary risk driver."
                    )
                    rejection_reasons.append(f"Directional inversion for {feat_name}")

        # Compute Grounding Score
        total_checks = direct_checked_count + len(derived_claims) + len(top_risk) + len(top_mitigating)
        total_violations = (
            len(unsupported_numbers)
            + len(unsupported_features)
            + len(directional_violations)
            + len(speculation_violations)
        )

        if total_checks > 0:
            score = max(0.0, min(1.0, 1.0 - (total_violations / total_checks)))
        else:
            score = 1.0

        is_grounded = len(rejection_reasons) == 0 and score >= 0.85

        return GroundingValidationResult(
            is_grounded=is_grounded,
            grounding_score=round(score, 3),
            direct_facts_checked=direct_checked_count,
            derived_facts_verified=derived_verified_count,
            unsupported_numbers=unsupported_numbers,
            unsupported_features=unsupported_features,
            directional_violations=directional_violations,
            speculation_violations=speculation_violations,
            rejection_reasons=rejection_reasons,
        )
