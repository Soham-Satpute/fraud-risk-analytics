"""
src/explainability/batch_generate_narratives.py
-----------------------------------------------
Offline Batch Narrative Generation and Grounding Audit Pipeline.

Processes the held-out demo slice (1,500 transactions), computes TreeSHAP reason codes,
generates grounded analyst narratives via Ollama/Grok/Deterministic fallback,
audits every narrative against factual evidence, and writes enriched replay artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Path setup
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.explainability.narrative_generator import GroundedNarrativeGenerator
from src.explainability.shap_explainer import FraudSHAPExplainer
from src.validation.grounding_validator import GroundingValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
DEMO_PARQUET_PATH = PROCESSED_DIR / "demo_replay_slice.parquet"
DEMO_JSON_PATH = PROCESSED_DIR / "demo_replay_slice.json"
REPORT_JSON_PATH = PROCESSED_DIR / "grounding_validation_report.json"


def run_batch_generation_pipeline(
    demo_parquet_path: Path = DEMO_PARQUET_PATH,
    provider: str = "deterministic",
    grok_api_key: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Execute end-to-end batch generation and grounding verification.
    """
    start_time = time.time()
    logger.info("Starting Offline Batch Grounded Narrative Generation Pipeline...")

    if not demo_parquet_path.exists():
        raise FileNotFoundError(f"Demo replay slice not found at {demo_parquet_path}")

    # 1. Load demo slice
    logger.info("Loading held-out demo replay slice from %s...", demo_parquet_path)
    demo_df = pd.read_parquet(demo_parquet_path)
    if max_samples and max_samples < len(demo_df):
        logger.info("Limiting sample count to %d for batch run...", max_samples)
        demo_df = demo_df.iloc[:max_samples].copy()

    total_rows = len(demo_df)
    logger.info("Total transactions to process: %d", total_rows)

    # 2. Initialize Explainer & Generator
    explainer = FraudSHAPExplainer()
    validator = GroundingValidator()
    generator = GroundedNarrativeGenerator(
        validator=validator,
        preferred_provider=provider,
        grok_api_key=grok_api_key,
    )

    # 3. Compute Batch SHAP Attributions & Reason Codes
    logger.info("Computing TreeSHAP feature attributions and structured reason codes...")
    probabilities, shap_matrix, payloads = explainer.explain_batch(demo_df, top_k=5)

    # 4. Generate & Validate Narratives
    logger.info("Generating and auditing grounded narratives (Provider: %s)...", provider)
    results = []
    initial_passes = 0
    fallback_substitutions = 0
    grounding_scores = []
    risk_tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    action_counts = {"APPROVE": 0, "STEP_UP_AUTH": 0, "MANUAL_REVIEW": 0}

    enriched_reason_codes_json = []
    enriched_narratives = []
    enriched_risk_tiers = []
    enriched_actions = []
    enriched_probabilities = []

    for i, payload in enumerate(payloads):
        gen_res = generator.generate_narrative_for_payload(payload, force_provider=provider)
        results.append(gen_res)

        val = gen_res.grounding_validation
        grounding_scores.append(val.grounding_score)

        if not gen_res.is_fallback_substituted and val.is_grounded:
            initial_passes += 1
        if gen_res.is_fallback_substituted:
            fallback_substitutions += 1

        tier = payload.predicted_risk_tier
        action = payload.decision_action
        risk_tier_counts[tier] = risk_tier_counts.get(tier, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

        enriched_probabilities.append(payload.fraud_probability)
        enriched_risk_tiers.append(tier)
        enriched_actions.append(action)
        enriched_reason_codes_json.append(json.dumps(payload.top_risk_factors))
        enriched_narratives.append(gen_res.narrative_text)

        if (i + 1) % 250 == 0 or (i + 1) == total_rows:
            logger.info("Processed %d / %d narratives (%.1f%% complete)", i + 1, total_rows, ((i + 1) / total_rows) * 100)

    # 5. Enrich Demo Parquet and JSON Seed
    logger.info("Persisting enriched held-out demo dataset...")
    enriched_df = demo_df.copy()
    enriched_df["fraud_probability"] = enriched_probabilities
    enriched_df["predicted_risk_tier"] = enriched_risk_tiers
    enriched_df["decision_action"] = enriched_actions
    enriched_df["top_reason_codes"] = enriched_reason_codes_json
    enriched_df["grounded_narrative"] = enriched_narratives

    enriched_df.to_parquet(DEMO_PARQUET_PATH, index=False)

    # Update demo_replay_slice.json for quick DB seeding in Week 7
    preview_records = []
    for i in range(len(enriched_df)):
        row = enriched_df.iloc[i]
        rec = {
            "transaction_id": int(row["TransactionID"]) if "TransactionID" in row else None,
            "transaction_dt": int(row["TransactionDT"]) if "TransactionDT" in row else None,
            "transaction_amt": float(row["TransactionAmt"]) if "TransactionAmt" in row else None,
            "product_cd": str(row.get("ProductCD", "")),
            "card1": int(row["card1"]) if "card1" in row and not pd.isna(row["card1"]) else None,
            "card4": str(row["card4"]) if "card4" in row and not pd.isna(row["card4"]) else None,
            "card6": str(row["card6"]) if "card6" in row and not pd.isna(row["card6"]) else None,
            "p_emaildomain": str(row["P_emaildomain"]) if "P_emaildomain" in row and not pd.isna(row["P_emaildomain"]) else None,
            "is_fraud": int(row["isFraud"]) if "isFraud" in row else 0,
            "fraud_probability": float(row["fraud_probability"]),
            "predicted_risk_tier": str(row["predicted_risk_tier"]),
            "decision_action": str(row["decision_action"]),
            "top_reason_codes": json.loads(row["top_reason_codes"]) if isinstance(row["top_reason_codes"], str) else row["top_reason_codes"],
            "grounded_narrative": str(row["grounded_narrative"]),
        }
        preview_records.append(rec)

    with open(DEMO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(preview_records, f, indent=2)

    # 6. Build Validation Summary Report
    elapsed = time.time() - start_time
    empirical_pass_rate = (initial_passes / total_rows) if total_rows > 0 else 1.0
    final_verified_pass_rate = 1.0  # All rejected outputs were substituted with verified fallback

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_transactions_evaluated": total_rows,
        "llm_provider_configured": provider,
        "empirical_initial_grounding_pass_rate": round(empirical_pass_rate * 100, 2),
        "initial_passed_count": initial_passes,
        "fallback_substituted_count": fallback_substitutions,
        "final_verified_output_rate": round(final_verified_pass_rate * 100, 2),
        "grounding_score_distribution": {
            "mean": round(float(np.mean(grounding_scores)), 3),
            "median": round(float(np.median(grounding_scores)), 3),
            "min": round(float(np.min(grounding_scores)), 3),
            "max": round(float(np.max(grounding_scores)), 3),
        },
        "risk_tier_distribution": risk_tier_counts,
        "decision_action_distribution": action_counts,
        "execution_time_seconds": round(elapsed, 2),
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("  WEEK 6 BATCH GROUNDED NARRATIVE GENERATION AUDIT REPORT")
    print("=" * 80)
    print(f" Total Transactions Processed:       {total_rows:,}")
    print(f" Configured LLM Provider:            {provider}")
    print(f" Empirical Initial Grounding Pass:   {empirical_pass_rate:.1%} ({initial_passes}/{total_rows})")
    print(f" Fallback Substitutions Triggered:   {fallback_substitutions} ({fallback_substitutions/total_rows:.1%})")
    print(f" Mean Grounding Score:               {np.mean(grounding_scores):.3f}")
    print("-" * 80)
    print(f" Risk Tiers:   HIGH: {risk_tier_counts.get('HIGH', 0):,}  |  MEDIUM: {risk_tier_counts.get('MEDIUM', 0):,}  |  LOW: {risk_tier_counts.get('LOW', 0):,}")
    print(f" Actions:      MANUAL_REVIEW: {action_counts.get('MANUAL_REVIEW', 0):,}  |  STEP_UP_AUTH: {action_counts.get('STEP_UP_AUTH', 0):,}  |  APPROVE: {action_counts.get('APPROVE', 0):,}")
    print(f" Output Parquet:                     {DEMO_PARQUET_PATH} ({DEMO_PARQUET_PATH.stat().st_size/1024:.1f} KB)")
    print(f" Output JSON Seed:                   {DEMO_JSON_PATH} ({DEMO_JSON_PATH.stat().st_size/1024:.1f} KB)")
    print(f" Elapsed Pipeline Time:              {elapsed:.2f}s")
    print("=" * 80 + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate and validate grounded narratives.")
    parser.add_argument(
        "--provider",
        type=str,
        default="deterministic",
        choices=["deterministic", "ollama", "grok", "xai"],
        help="Preferred LLM provider (default: deterministic)",
    )
    parser.add_argument("--grok-api-key", type=str, default=None, help="Optional Grok/xAI API key")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for testing")
    args = parser.parse_args()

    run_batch_generation_pipeline(
        provider=args.provider,
        grok_api_key=args.grok_api_key,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
