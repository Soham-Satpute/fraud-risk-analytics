"""
src/explainability/reason_codes.py
----------------------------------
Reason Code Extraction, Collinear V-Feature Consolidation, Domain Descriptor Mapping,
and Business Action Policy Resolver.

Transforms raw TreeSHAP feature attributions into stakeholder-interpretable,
grounded reason codes aligned with the operational decision policy established in Week 5.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Operational Business Action Policy (Week 5 Threshold Boundaries)
# -----------------------------------------------------------------------------
# Based on Week 5 LightGBM operational evaluation:
# - Low Risk: p < 0.10 -> APPROVE (Normal standard processing)
# - Medium Risk: 0.10 <= p < 0.35 -> STEP_UP_AUTH (Additional verification / 3DS / OTP)
# - High Risk: p >= 0.35 -> MANUAL_REVIEW (Prioritized fraud analyst review queue)
DEFAULT_THRESHOLD_MEDIUM = 0.10
DEFAULT_THRESHOLD_HIGH = 0.35


@dataclass(frozen=True)
class BusinessDecisionPolicy:
    """Predefined operational policy mapping model probability to action."""
    threshold_medium: float = DEFAULT_THRESHOLD_MEDIUM
    threshold_high: float = DEFAULT_THRESHOLD_HIGH

    def resolve_tier_and_action(self, probability: float) -> tuple[str, str, str]:
        """
        Determine Risk Tier, Decision Action, and Recommended Workflow.

        Returns:
            tuple: (risk_tier, decision_action, recommended_workflow)
        """
        if probability >= self.threshold_high:
            return (
                "HIGH",
                "MANUAL_REVIEW",
                "Route to prioritized manual fraud investigation queue before fulfillment.",
            )
        elif probability >= self.threshold_medium:
            return (
                "MEDIUM",
                "STEP_UP_AUTH",
                "Trigger step-up authentication (3D-Secure, OTP, or identity verification).",
            )
        else:
            return (
                "LOW",
                "APPROVE",
                "Approve transaction for automated standard straight-through processing.",
            )


# -----------------------------------------------------------------------------
# 2. Human-Readable Domain Feature Descriptions & Templates
# -----------------------------------------------------------------------------
FEATURE_DOMAIN_MAP: dict[str, dict[str, Any]] = {
    # Engineered features
    "amt_zscore_card1": {
        "display_name": "Card Amount Deviation (Z-Score)",
        "category": "AMOUNT_ANOMALY",
        "description": "Standardized deviation of transaction amount from card historical mean",
        "unit": "sigma",
    },
    "amt_diff_mean_card1": {
        "display_name": "Card Amount Difference from Mean",
        "category": "AMOUNT_ANOMALY",
        "description": "Dollar difference between transaction amount and card average",
        "unit": "$",
    },
    "amt_ratio_mean_card1": {
        "display_name": "Card Amount to Average Ratio",
        "category": "AMOUNT_ANOMALY",
        "description": "Ratio of transaction amount relative to card average spending",
        "unit": "ratio",
    },
    "amt_zscore_card1_addr1": {
        "display_name": "Card & Billing Location Amount Z-Score",
        "category": "AMOUNT_ANOMALY",
        "description": "Amount deviation within the card and billing region cluster",
        "unit": "sigma",
    },
    "amt_zscore_email": {
        "display_name": "Email Domain Amount Z-Score",
        "category": "AMOUNT_ANOMALY",
        "description": "Amount deviation relative to the email domain historical baseline",
        "unit": "sigma",
    },
    "log_TransactionAmt": {
        "display_name": "Log Transaction Amount",
        "category": "AMOUNT_ANOMALY",
        "description": "Log-scale magnitude of transaction amount",
        "unit": "log($)",
    },
    "hour_sin": {
        "display_name": "Diurnal Cycle (Sine Component)",
        "category": "TEMPORAL_PATTERN",
        "description": "Cyclical time-of-day feature capturing diurnal attack windows",
        "unit": "cyclical",
    },
    "hour_cos": {
        "display_name": "Diurnal Cycle (Cosine Component)",
        "category": "TEMPORAL_PATTERN",
        "description": "Cyclical time-of-day feature capturing diurnal attack windows",
        "unit": "cyclical",
    },
    "dow_sin": {
        "display_name": "Weekly Cycle (Sine Component)",
        "category": "TEMPORAL_PATTERN",
        "description": "Cyclical day-of-week feature",
        "unit": "cyclical",
    },
    "dow_cos": {
        "display_name": "Weekly Cycle (Cosine Component)",
        "category": "TEMPORAL_PATTERN",
        "description": "Cyclical day-of-week feature",
        "unit": "cyclical",
    },
    "freq_card1": {
        "display_name": "Card Identifier Frequency",
        "category": "VELOCITY_AND_FREQUENCY",
        "description": "Historical appearance frequency of primary card number proxy",
        "unit": "count_fraction",
    },
    "freq_card2": {
        "display_name": "Card Sub-Type Frequency",
        "category": "VELOCITY_AND_FREQUENCY",
        "description": "Historical appearance frequency of card sub-identifier",
        "unit": "count_fraction",
    },
    "freq_addr1": {
        "display_name": "Billing Region Frequency",
        "category": "LOCATION_FREQUENCY",
        "description": "Historical volume frequency of billing address region",
        "unit": "count_fraction",
    },
    "freq_ProductCD": {
        "display_name": "Product Category Frequency",
        "category": "PRODUCT_FREQUENCY",
        "description": "Historical volume frequency of product category code",
        "unit": "count_fraction",
    },
    "freq_R_emaildomain": {
        "display_name": "Recipient Email Domain Frequency",
        "category": "EMAIL_FREQUENCY",
        "description": "Historical frequency of recipient email domain in transactions",
        "unit": "count_fraction",
    },
    "email_match_flag": {
        "display_name": "Payer-Recipient Email Match Flag",
        "category": "IDENTITY_CONSISTENCY",
        "description": "Indicator whether purchaser and recipient email domains are identical",
        "unit": "binary",
    },
    "null_P_email": {
        "display_name": "Missing Purchaser Email Flag",
        "category": "IDENTITY_MISSINGNESS",
        "description": "Indicator of missing purchaser email domain",
        "unit": "binary",
    },
    "null_R_email": {
        "display_name": "Missing Recipient Email Flag",
        "category": "IDENTITY_MISSINGNESS",
        "description": "Indicator of missing recipient email domain",
        "unit": "binary",
    },
    # Core raw features
    "TransactionAmt": {
        "display_name": "Transaction Amount",
        "category": "AMOUNT",
        "description": "Raw payment amount in USD",
        "unit": "$",
    },
    "ProductCD": {
        "display_name": "Product Code Channel",
        "category": "PRODUCT",
        "description": "Product / transaction channel classification (e.g. W, C, R, H)",
        "unit": "category",
    },
    "card1": {
        "display_name": "Card Number Issuer Proxy",
        "category": "PAYMENT_METHOD",
        "description": "Primary card identification code",
        "unit": "id",
    },
    "card2": {
        "display_name": "Card Sub-Type Code",
        "category": "PAYMENT_METHOD",
        "description": "Secondary card classification code",
        "unit": "id",
    },
    "card3": {
        "display_name": "Card Country Code",
        "category": "PAYMENT_METHOD",
        "description": "Card issuing country identifier",
        "unit": "id",
    },
    "card4": {
        "display_name": "Card Brand Network",
        "category": "PAYMENT_METHOD",
        "description": "Payment card network (e.g. Visa, Mastercard, Amex, Discover)",
        "unit": "category",
    },
    "card6": {
        "display_name": "Card Funding Type",
        "category": "PAYMENT_METHOD",
        "description": "Card account funding type (e.g. credit, debit)",
        "unit": "category",
    },
    "addr1": {
        "display_name": "Billing Region Code",
        "category": "LOCATION",
        "description": "Billing zip/postal region code",
        "unit": "id",
    },
    "addr2": {
        "display_name": "Billing Country Code",
        "category": "LOCATION",
        "description": "Billing country code identifier",
        "unit": "id",
    },
    "P_emaildomain": {
        "display_name": "Purchaser Email Domain",
        "category": "IDENTITY",
        "description": "Email domain of purchaser",
        "unit": "domain",
    },
    "R_emaildomain": {
        "display_name": "Recipient Email Domain",
        "category": "IDENTITY",
        "description": "Email domain of recipient (e.g. transfer/remittance flows)",
        "unit": "domain",
    },
    "C1": {
        "display_name": "Transaction Velocity Count Proxy (C1)",
        "category": "VELOCITY",
        "description": "Historical transaction count velocity proxy associated with card/entity",
        "unit": "count",
    },
    "C2": {
        "display_name": "Transaction Velocity Count Proxy (C2)",
        "category": "VELOCITY",
        "description": "Related transaction count velocity proxy",
        "unit": "count",
    },
    "D1": {
        "display_name": "Days Since Prior Transaction (D1)",
        "category": "TEMPORAL_RECENCY",
        "description": "Elapsed days since the last recorded transaction on this card/entity",
        "unit": "days",
    },
    "D2": {
        "display_name": "Days Since Previous Event (D2)",
        "category": "TEMPORAL_RECENCY",
        "description": "Time delta in days from prior event",
        "unit": "days",
    },
}

# -----------------------------------------------------------------------------
# 3. Collinear V-Feature Clusters (from Week 1 Audit: |r| >= 0.98)
# -----------------------------------------------------------------------------
# Maps redundant collinear V-features into representative cluster concepts
V_COLLINEAR_CLUSTERS: dict[str, dict[str, Any]] = {
    "V95_V101_V279_V293": {
        "members": {"V95", "V96", "V97", "V101", "V102", "V103", "V132", "V133", "V134", "V279", "V280", "V293", "V295", "V316", "V318"},
        "cluster_name": "Payment Transaction Volume & Count Cluster (V95-V103/V279-V295)",
        "display_name": "Payment Activity Volume Cluster",
        "description": "Aggregated payment velocity and volume counters across 24h-30d windows",
    },
    "V167_V177_V211": {
        "members": {"V167", "V168", "V177", "V178", "V179", "V211", "V212", "V213"},
        "cluster_name": "Card Verification Activity Cluster (V167-V179/V211-V213)",
        "display_name": "Card Verification Activity Cluster",
        "description": "Frequency of card verification attempts and associated device signals",
    },
    "V217_V233": {
        "members": {"V217", "V219", "V231", "V232", "V233"},
        "cluster_name": "Digital Checkout Device Velocity Cluster (V217-V233)",
        "display_name": "Device Checkout Velocity Cluster",
        "description": "Device and browser event velocity during digital checkout flows",
    },
    "V240_V244": {
        "members": {"V240", "V241", "V242", "V244", "V250", "V251"},
        "cluster_name": "Digital Channel Signature Cluster (V240-V251)",
        "display_name": "Digital Channel Signature Cluster",
        "description": "Digital channel security protocol and behavioral match indicators",
    },
    "V257_V258": {
        "members": {"V257", "V258", "V246", "V200", "V201"},
        "cluster_name": "High-Risk Channel Indicator Cluster (V257-V258/V200-V201)",
        "display_name": "Adversarial Risk Pattern Cluster",
        "description": "High-correlation risk indicators associated with coordinated digital attacks",
    },
}

# Reverse lookup from member V-feature -> cluster metadata
_V_MEMBER_TO_CLUSTER: dict[str, dict[str, Any]] = {}
for cluster_key, cluster_info in V_COLLINEAR_CLUSTERS.items():
    for member in cluster_info["members"]:
        _V_MEMBER_TO_CLUSTER[member] = cluster_info


# -----------------------------------------------------------------------------
# 4. Reason Code Data Structures
# -----------------------------------------------------------------------------
@dataclass
class ReasonCode:
    """Individual structured feature attribution reason code."""
    feature: str
    display_name: str
    feature_value: Any
    shap_value: float
    contribution_pct: float
    direction: str  # "INCREASES_RISK" | "REDUCES_RISK"
    category: str
    description: str
    is_collinear_cluster: bool = False
    cluster_members: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionExplanationPayload:
    """Full structured explanation payload for a single transaction."""
    transaction_id: int | None
    fraud_probability: float
    predicted_risk_tier: str
    decision_action: str
    recommended_workflow: str
    base_value_log_odds: float
    model_version: str
    top_risk_factors: list[dict[str, Any]]
    top_mitigating_factors: list[dict[str, Any]]
    context_attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# 5. Reason Code Extraction Engine
# -----------------------------------------------------------------------------
class ReasonCodeEngine:
    """
    Extracts and formats structured reason codes from raw TreeSHAP values,
    consolidating collinear feature clusters and formatting human-readable context.
    """

    def __init__(self, policy: BusinessDecisionPolicy | None = None) -> None:
        self.policy = policy or BusinessDecisionPolicy()

    def get_feature_display_info(self, feature_name: str) -> dict[str, Any]:
        """Lookup human-readable info or generate clean fallback."""
        if feature_name in FEATURE_DOMAIN_MAP:
            return FEATURE_DOMAIN_MAP[feature_name]

        # Check for V-feature cluster
        if feature_name in _V_MEMBER_TO_CLUSTER:
            cl = _V_MEMBER_TO_CLUSTER[feature_name]
            return {
                "display_name": f"{cl['display_name']} ({feature_name})",
                "category": "V_CLUSTER",
                "description": cl["description"],
                "unit": "index",
            }

        if feature_name.startswith("V"):
            return {
                "display_name": f"Transaction Behavioral Feature ({feature_name})",
                "category": "BEHAVIORAL_V",
                "description": f"Vesta behavioral attribute {feature_name}",
                "unit": "numeric",
            }
        elif feature_name.startswith("C"):
            return {
                "display_name": f"Transaction Velocity Counter ({feature_name})",
                "category": "VELOCITY_C",
                "description": f"Historical count velocity metric {feature_name}",
                "unit": "count",
            }
        elif feature_name.startswith("D"):
            return {
                "display_name": f"Time-Delta Recency Feature ({feature_name})",
                "category": "TIME_DELTA_D",
                "description": f"Days elapsed delta feature {feature_name}",
                "unit": "days",
            }
        elif feature_name.startswith("M"):
            return {
                "display_name": f"Identity Verification Match ({feature_name})",
                "category": "MATCH_STATUS_M",
                "description": f"Address / cardholder verification indicator {feature_name}",
                "unit": "status",
            }

        return {
            "display_name": feature_name.replace("_", " ").title(),
            "category": "GENERAL",
            "description": f"Model feature {feature_name}",
            "unit": "value",
        }

    def consolidate_and_extract_reason_codes(
        self,
        feature_names: list[str],
        feature_values: list[Any] | np.ndarray,
        shap_values: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[ReasonCode], list[ReasonCode]]:
        """
        Extract top-k positive (risk-increasing) and negative (risk-mitigating) reason codes,
        consolidating collinear V-feature clusters so redundant features are merged.
        """
        raw_items: list[dict[str, Any]] = []
        for name, val, sv in zip(feature_names, feature_values, shap_values):
            # Clean value representation for serialization
            if isinstance(val, (np.floating, float)):
                clean_val: Any = round(float(val), 4) if not np.isnan(val) else None
            elif isinstance(val, (np.integer, int)):
                clean_val = int(val)
            else:
                clean_val = str(val) if val is not None and not pd.isna(val) else None

            raw_items.append({
                "feature": name,
                "value": clean_val,
                "shap_value": float(sv),
            })

        # Separate positive (risk increasing) and negative (risk reducing)
        pos_items = [item for item in raw_items if item["shap_value"] > 0]
        neg_items = [item for item in raw_items if item["shap_value"] < 0]

        # Sort by absolute magnitude
        pos_items.sort(key=lambda x: x["shap_value"], reverse=True)
        neg_items.sort(key=lambda x: x["shap_value"])  # Most negative first

        def _consolidate_list(items: list[dict[str, Any]], direction: str) -> list[ReasonCode]:
            consolidated: list[ReasonCode] = []
            seen_clusters: set[str] = set()
            total_abs_shap = sum(abs(x["shap_value"]) for x in items) if items else 1.0

            for item in items:
                feat = item["feature"]
                sv = item["shap_value"]
                val = item["value"]

                # Check if feature belongs to a known collinear V cluster
                if feat in _V_MEMBER_TO_CLUSTER:
                    cl = _V_MEMBER_TO_CLUSTER[feat]
                    c_name = cl["cluster_name"]

                    if c_name in seen_clusters:
                        # Cluster already represented by a higher-magnitude member; skip redundant duplicate
                        continue
                    seen_clusters.add(c_name)

                    contrib_pct = round((abs(sv) / total_abs_shap) * 100, 2)
                    consolidated.append(ReasonCode(
                        feature=feat,
                        display_name=cl["display_name"],
                        feature_value=val,
                        shap_value=round(sv, 4),
                        contribution_pct=contrib_pct,
                        direction=direction,
                        category="V_COLLINEAR_CLUSTER",
                        description=f"{cl['description']} (represented by {feat}={val})",
                        is_collinear_cluster=True,
                        cluster_members=list(cl["members"]),
                    ))
                else:
                    info = self.get_feature_display_info(feat)
                    contrib_pct = round((abs(sv) / total_abs_shap) * 100, 2)

                    # Contextual description formatting
                    desc = info["description"]
                    if "sigma" in info.get("unit", "") and isinstance(val, (int, float)):
                        desc = f"{info['description']} (Observed: {val:+.2f}σ)"
                    elif info.get("unit") == "$" and isinstance(val, (int, float)):
                        desc = f"{info['description']} (Observed: ${val:,.2f})"

                    consolidated.append(ReasonCode(
                        feature=feat,
                        display_name=info["display_name"],
                        feature_value=val,
                        shap_value=round(sv, 4),
                        contribution_pct=contrib_pct,
                        direction=direction,
                        category=info.get("category", "GENERAL"),
                        description=desc,
                        is_collinear_cluster=False,
                    ))

                if len(consolidated) >= top_k:
                    break

            return consolidated

        top_risk = _consolidate_list(pos_items, "INCREASES_RISK")
        top_mitigating = _consolidate_list(neg_items, "REDUCES_RISK")

        return top_risk, top_mitigating

    def build_explanation_payload(
        self,
        transaction_id: int | None,
        probability: float,
        base_value_log_odds: float,
        feature_names: list[str],
        feature_values: list[Any] | np.ndarray,
        shap_values: np.ndarray,
        raw_context: dict[str, Any] | None = None,
        model_version: str = "v1.0.0-champion-lgbm",
        top_k: int = 5,
    ) -> TransactionExplanationPayload:
        """
        Construct complete structured explanation payload.
        """
        prob = round(float(probability), 4)
        risk_tier, decision_action, recommended_workflow = self.policy.resolve_tier_and_action(prob)

        top_risk, top_mitigating = self.consolidate_and_extract_reason_codes(
            feature_names=feature_names,
            feature_values=feature_values,
            shap_values=shap_values,
            top_k=top_k,
        )

        context = raw_context or {}

        return TransactionExplanationPayload(
            transaction_id=transaction_id,
            fraud_probability=prob,
            predicted_risk_tier=risk_tier,
            decision_action=decision_action,
            recommended_workflow=recommended_workflow,
            base_value_log_odds=round(float(base_value_log_odds), 4),
            model_version=model_version,
            top_risk_factors=[rc.to_dict() for rc in top_risk],
            top_mitigating_factors=[rc.to_dict() for rc in top_mitigating],
            context_attributes=context,
        )
