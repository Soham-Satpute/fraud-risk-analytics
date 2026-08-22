"""
src/data/seed_supabase.py
-------------------------
Database Migration and Seed CLI for Supabase PostgreSQL Serving Layer.

Supports:
  1. Executing DDL from sql/schema.sql to create tables and partial indexes.
  2. Seeding model_runs table with baseline and champion benchmarks from models/model_metrics.json.
  3. Bulk loading 1,500 held-out test transactions and grounded narratives from data/processed/demo_replay_slice.json into demo_replay.
  4. Simulating batch inference replay to populate predictions table for Power BI and Week 8 analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_supabase")


def get_db_connection(db_url: str | None = None):
    """Establish direct connection to PostgreSQL database."""
    url = db_url or settings.DATABASE_URL or settings.SUPABASE_DB_URL
    if not url:
        raise ValueError(
            "No database connection string provided. Please set DATABASE_URL or SUPABASE_DB_URL environment variable."
        )
    import psycopg2
    return psycopg2.connect(url, connect_timeout=10)


def apply_schema(conn, schema_path: Path | None = None) -> None:
    """Execute schema DDL script."""
    path = schema_path or (_PROJECT_ROOT / "sql" / "schema.sql")
    logger.info("Applying schema from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    with conn.cursor() as cur:
        cur.execute(sql_content)
    conn.commit()
    logger.info("Schema DDL applied successfully.")


def seed_model_runs(conn, metrics_path: Path | None = None) -> int:
    """Seed model_runs table with benchmark metrics."""
    path = metrics_path or Path(settings.METRICS_PATH)
    logger.info("Seeding model_runs from %s...", path)
    if not path.exists():
        logger.warning("Metrics file %s does not exist. Skipping model_runs seed.", path)
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    # Baseline
    baseline = data.get("baseline_logistic_regression", {})
    if baseline:
        runs.append({
            "run_id": "run-baseline-lr-v1",
            "model_name": "Logistic Regression Baseline",
            "model_version": "v1.0.0",
            "validation_strategy": "Temporal Split (TransactionDT <= 12,192,854)",
            "operating_threshold": 0.50,
            "total_evaluated": 118108,
            "pr_auc": baseline.get("pr_auc", 0.2746),
            "roc_auc": baseline.get("roc_auc", 0.8092),
            "recall_at_1pct_fpr": baseline.get("recall_at_1pct_fpr", 0.1508),
            "recall_at_5pct_fpr": baseline.get("recall_at_5pct_fpr", 0.4176),
            "metadata": json.dumps({"class_weight": "balanced", "scaler": "StandardScaler"}),
        })

    # Champion
    champ = data.get("champion_lightgbm", {})
    if champ:
        runs.append({
            "run_id": "run-champion-lgbm-v1",
            "model_name": "Champion LightGBM Classifier",
            "model_version": "v1.0.0",
            "validation_strategy": "Temporal Split (TransactionDT <= 12,192,854)",
            "operating_threshold": 0.35,
            "total_evaluated": 118108,
            "pr_auc": champ.get("pr_auc", 0.5441),
            "roc_auc": champ.get("roc_auc", 0.9035),
            "recall_at_1pct_fpr": champ.get("recall_at_1pct_fpr", 0.4663),
            "recall_at_5pct_fpr": champ.get("recall_at_5pct_fpr", 0.6595),
            "metadata": json.dumps({"scale_pos_weight": 27.46, "num_trees": 200}),
        })

    with conn.cursor() as cur:
        query = """
            INSERT INTO model_runs (
                run_id, model_name, model_version, validation_strategy,
                operating_threshold, total_evaluated, pr_auc, roc_auc,
                recall_at_1pct_fpr, recall_at_5pct_fpr, metadata
            ) VALUES (
                %(run_id)s, %(model_name)s, %(model_version)s, %(validation_strategy)s,
                %(operating_threshold)s, %(total_evaluated)s, %(pr_auc)s, %(roc_auc)s,
                %(recall_at_1pct_fpr)s, %(recall_at_5pct_fpr)s, %(metadata)s
            ) ON CONFLICT (run_id) DO UPDATE SET
                pr_auc = EXCLUDED.pr_auc,
                roc_auc = EXCLUDED.roc_auc,
                recall_at_1pct_fpr = EXCLUDED.recall_at_1pct_fpr,
                recall_at_5pct_fpr = EXCLUDED.recall_at_5pct_fpr;
        """
        for r in runs:
            cur.execute(query, r)
    conn.commit()
    logger.info("Successfully seeded %d model run records.", len(runs))
    return len(runs)


def seed_demo_replay(conn, demo_path: Path | None = None) -> int:
    """Bulk insert 1,500 held-out test transactions into demo_replay."""
    path = demo_path or Path(settings.DEMO_REPLAY_PATH)
    logger.info("Seeding demo_replay from %s...", path)
    if not path.exists():
        logger.warning("Demo replay file %s does not exist. Skipping demo_replay seed.", path)
        return 0

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Found %d records in %s. Inserting in batches...", len(records), path.name)

    query = """
        INSERT INTO demo_replay (
            transaction_id, transaction_dt, transaction_amt, product_cd,
            card1, card4, card6, p_emaildomain, is_fraud,
            feature_payload, grounded_narrative
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (transaction_id) DO UPDATE SET
            grounded_narrative = EXCLUDED.grounded_narrative;
    """

    inserted = 0
    batch_size = 100
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            args_list = [
                (
                    r["transaction_id"],
                    r["transaction_dt"],
                    float(r["transaction_amt"]),
                    r.get("product_cd", "W"),
                    r.get("card1", 0),
                    r.get("card4"),
                    r.get("card6"),
                    r.get("p_emaildomain"),
                    int(r.get("is_fraud", 0)),
                    json.dumps(r.get("top_reason_codes", [])),
                    r.get("grounded_narrative"),
                )
                for r in batch
            ]
            cur.executemany(query, args_list)
            inserted += len(batch)
    conn.commit()
    logger.info("Successfully seeded %d demo replay records.", inserted)
    return inserted


def replay_predictions_into_db(conn, demo_path: Path | None = None) -> int:
    """
    Replay held-out demo transactions to populate predictions table for Power BI and Week 8 analysis.
    """
    path = demo_path or Path(settings.DEMO_REPLAY_PATH)
    logger.info("Replaying demo transactions into predictions table from %s...", path)
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    query = """
        INSERT INTO predictions (
            prediction_id, transaction_id, transaction_dt, transaction_amt,
            fraud_probability, predicted_risk_tier, decision_action,
            actual_label, top_reason_codes, grounded_narrative,
            model_version, latency_ms, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (prediction_id) DO NOTHING;
    """

    batch_size = 100
    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            args_list = [
                (
                    str(uuid.uuid4()),
                    r["transaction_id"],
                    r["transaction_dt"],
                    float(r["transaction_amt"]),
                    float(r.get("fraud_probability", 0.0)),
                    r.get("predicted_risk_tier", "LOW"),
                    r.get("decision_action", "APPROVE"),
                    int(r.get("is_fraud", 0)),
                    json.dumps(r.get("top_reason_codes", [])),
                    r.get("grounded_narrative"),
                    "v1.0.0",
                    42.5,
                    datetime.now(timezone.utc).isoformat(),
                )
                for r in batch
            ]
            cur.executemany(query, args_list)
            inserted += len(batch)
    conn.commit()
    logger.info("Successfully replayed %d transactions into predictions table.", inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Supabase PostgreSQL Migration & Seeding CLI")
    parser.add_argument("--schema", action="store_true", help="Apply schema DDL")
    parser.add_argument("--seed-model-runs", action="store_true", help="Seed model_runs table")
    parser.add_argument("--seed-demo-replay", action="store_true", help="Seed demo_replay table")
    parser.add_argument("--replay-predictions", action="store_true", help="Replay demo slice into predictions")
    parser.add_argument("--all", action="store_true", help="Run all migration and seed steps")
    parser.add_argument("--dry-run", action="store_true", help="Verify files without connecting to DB")
    parser.add_argument("--db-url", type=str, default=None, help="Explicit PostgreSQL connection URL")

    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Schema file: %s (exists=%s)", _PROJECT_ROOT / "sql" / "schema.sql", (_PROJECT_ROOT / "sql" / "schema.sql").exists())
        logger.info("Metrics file: %s (exists=%s)", settings.METRICS_PATH, Path(settings.METRICS_PATH).exists())
        logger.info("Demo replay file: %s (exists=%s)", settings.DEMO_REPLAY_PATH, Path(settings.DEMO_REPLAY_PATH).exists())
        logger.info("Dry run check completed successfully.")
        return

    db_url = args.db_url or settings.DATABASE_URL or settings.SUPABASE_DB_URL
    if not db_url:
        logger.warning(
            "No DATABASE_URL or SUPABASE_DB_URL set. "
            "To connect to Supabase, provide --db-url or set the environment variable. "
            "API will operate using resilient In-Memory fallback."
        )
        return

    try:
        conn = get_db_connection(db_url)
        logger.info("Connected to PostgreSQL database.")
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL database: %s", str(e))
        sys.exit(1)

    try:
        if args.all or args.schema:
            apply_schema(conn)
        if args.all or args.seed_model_runs:
            seed_model_runs(conn)
        if args.all or args.seed_demo_replay:
            seed_demo_replay(conn)
        if args.all or args.replay_predictions:
            replay_predictions_into_db(conn)
        logger.info("All requested database operations completed successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
