"""
dashboard/export_analytics_extracts.py
--------------------------------------
Exports SQL and offline inference analytics into structured CSV datasets in dashboard/data/.
Enables seamless Power BI dashboard development and offline visualization without requiring
live cloud PostgreSQL access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_extracts")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
DASHBOARD_DATA_DIR = _PROJECT_ROOT / "dashboard" / "data"
DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)


def export_analytics_csvs() -> None:
    """Extract and write analytical dataframes to dashboard/data/."""
    demo_json_path = DATA_PROCESSED_DIR / "demo_replay_slice.json"
    if not demo_json_path.exists():
        logger.error("Source demo slice %s not found.", demo_json_path)
        return

    with open(demo_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Loaded %d held-out test transactions for dashboard extracts.", len(records))

    # 1. Predictions Summary Dataset
    rows = []
    for r in records:
        top_reasons = r.get("top_reason_codes", [])
        top_feature_name = top_reasons[0].get("display_name") if top_reasons else "N/A"
        top_feature_shap = top_reasons[0].get("shap_value") if top_reasons else 0.0

        rows.append({
            "transaction_id": r["transaction_id"],
            "transaction_dt": r["transaction_dt"],
            "transaction_amt": float(r["transaction_amt"]),
            "product_cd": r.get("product_cd", "W"),
            "card1": r.get("card1", 0),
            "card4": r.get("card4", "unknown"),
            "card6": r.get("card6", "unknown"),
            "p_emaildomain": r.get("p_emaildomain", "unknown"),
            "is_fraud": int(r.get("is_fraud", 0)),
            "fraud_probability": float(r.get("fraud_probability", 0.0)),
            "predicted_risk_tier": r.get("predicted_risk_tier", "LOW"),
            "decision_action": r.get("decision_action", "APPROVE"),
            "top_risk_feature": top_feature_name,
            "top_feature_shap": top_feature_shap,
        })

    df_preds = pd.DataFrame(rows)
    df_preds.to_csv(DASHBOARD_DATA_DIR / "predictions_summary.csv", index=False)
    logger.info("Exported predictions_summary.csv (%d rows)", len(df_preds))

    # 2. Score Distribution Deciles
    df_preds["score_decile"] = pd.cut(
        df_preds["fraud_probability"],
        bins=np.linspace(0.0, 1.0, 11),
        labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)],
        include_lowest=True,
    )
    deciles = df_preds.groupby("score_decile", observed=False).agg(
        total_transactions=("transaction_id", "count"),
        fraud_transactions=("is_fraud", "sum"),
        mean_probability=("fraud_probability", "mean"),
    ).reset_index()
    deciles["empirical_fraud_rate_pct"] = (
        deciles["fraud_transactions"] / deciles["total_transactions"].replace(0, np.nan) * 100.0
    ).fillna(0.0).round(2)
    deciles.to_csv(DASHBOARD_DATA_DIR / "score_distribution.csv", index=False)
    logger.info("Exported score_distribution.csv (%d deciles)", len(deciles))

    # 3. Operational vs Evaluation Metrics Summary
    total_tx = len(df_preds)
    tier_counts = df_preds["predicted_risk_tier"].value_counts().to_dict()
    op_metrics = pd.DataFrame([{
        "metric_category": "Operational Monitoring (Unlabeled)",
        "total_transactions_logged": total_tx,
        "low_risk_volume": tier_counts.get("LOW", 0),
        "low_risk_pct": round(tier_counts.get("LOW", 0) / total_tx * 100.0, 2),
        "medium_risk_volume": tier_counts.get("MEDIUM", 0),
        "medium_risk_pct": round(tier_counts.get("MEDIUM", 0) / total_tx * 100.0, 2),
        "high_risk_volume": tier_counts.get("HIGH", 0),
        "high_risk_pct": round(tier_counts.get("HIGH", 0) / total_tx * 100.0, 2),
        "avg_predicted_probability": round(df_preds["fraud_probability"].mean(), 4),
    }])
    op_metrics.to_csv(DASHBOARD_DATA_DIR / "operational_metrics.csv", index=False)
    logger.info("Exported operational_metrics.csv")

    # 4. Labeled Evaluation Performance Benchmark
    tp = len(df_preds[(df_preds["is_fraud"] == 1) & (df_preds["predicted_risk_tier"] == "HIGH")])
    fp = len(df_preds[(df_preds["is_fraud"] == 0) & (df_preds["predicted_risk_tier"] == "HIGH")])
    tn = len(df_preds[(df_preds["is_fraud"] == 0) & (df_preds["predicted_risk_tier"] != "HIGH")])
    fn = len(df_preds[(df_preds["is_fraud"] == 1) & (df_preds["predicted_risk_tier"] != "HIGH")])

    eval_metrics = pd.DataFrame([{
        "dataset_description": "Held-Out Test Set Labeled Benchmark",
        "total_evaluated": total_tx,
        "actual_fraud_count": int(df_preds["is_fraud"].sum()),
        "actual_fraud_rate_pct": round(df_preds["is_fraud"].mean() * 100.0, 2),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision_pct": round(tp / (tp + fp) * 100.0, 2) if (tp + fp) > 0 else 0.0,
        "recall_pct": round(tp / (tp + fn) * 100.0, 2) if (tp + fn) > 0 else 0.0,
        "false_positive_rate_pct": round(fp / (fp + tn) * 100.0, 2) if (fp + tn) > 0 else 0.0,
    }])
    eval_metrics.to_csv(DASHBOARD_DATA_DIR / "evaluation_metrics.csv", index=False)
    logger.info("Exported evaluation_metrics.csv")

    # 5. High-Risk Review Queue Extract
    df_high_risk = df_preds[df_preds["predicted_risk_tier"] == "HIGH"].sort_values(
        by="fraud_probability", ascending=False
    )
    df_high_risk.to_csv(DASHBOARD_DATA_DIR / "high_risk_review_queue.csv", index=False)
    logger.info("Exported high_risk_review_queue.csv (%d items)", len(df_high_risk))

    logger.info("All Power BI static analytics extracts generated successfully in %s", DASHBOARD_DATA_DIR)


if __name__ == "__main__":
    export_analytics_csvs()
