"""
tests/test_api.py
-----------------
Automated test suite for FastAPI serving layer, inference endpoints,
security guards, latency benchmarking, and monitoring endpoints.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.config import settings
from api.db import db_manager
from api.main import create_app

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_JSON_PATH = _PROJECT_ROOT / "data" / "processed" / "demo_replay_slice.json"


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized lifespan context."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_transactions() -> list[dict[str, Any]]:
    """Load sample held-out test transactions from demo slice."""
    if DEMO_JSON_PATH.exists():
        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)[:10]
    return [
        {
            "TransactionID": 3459526,
            "TransactionDT": 12198174,
            "TransactionAmt": 117.0,
            "ProductCD": "W",
            "card1": 4436,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "is_fraud": 0,
        }
    ]


# -----------------------------------------------------------------------------
# 1. System Health & Metadata Tests
# -----------------------------------------------------------------------------
def test_root_endpoint(client: TestClient):
    """Verify root endpoint returns API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["app_name"] == settings.APP_NAME
    assert data["version"] == settings.APP_VERSION


def test_health_endpoint(client: TestClient):
    """Verify health endpoint confirms model and pipeline readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True
    assert "benchmark_inference_latency_ms" in data


# -----------------------------------------------------------------------------
# 2. Single Transaction Scoring Tests
# -----------------------------------------------------------------------------
def test_predict_single_transaction(client: TestClient, sample_transactions: list[dict[str, Any]]):
    """Verify single transaction scoring, TreeSHAP reason codes, and measured latency."""
    sample = sample_transactions[0]
    payload = {
        "TransactionID": sample.get("transaction_id", 3459526),
        "TransactionDT": sample.get("transaction_dt", 12198174),
        "TransactionAmt": float(sample.get("transaction_amt", 117.0)),
        "ProductCD": sample.get("product_cd", "W"),
        "card1": sample.get("card1", 4436),
        "card4": sample.get("card4", "visa"),
        "card6": sample.get("card6", "debit"),
        "P_emaildomain": sample.get("p_emaildomain", "gmail.com"),
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # Probability bounds
    assert 0.0 <= data["predicted_probability"] <= 1.0
    assert data["predicted_risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert data["decision_action"] in ("APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW")
    assert isinstance(data["recommended_workflow"], str)

    # Reason codes structure
    assert len(data["top_reason_codes"]) > 0
    top_rc = data["top_reason_codes"][0]
    assert "feature" in top_rc
    assert "display_name" in top_rc
    assert "shap_value" in top_rc
    assert top_rc["direction"] == "INCREASES_RISK"
    assert "contribution_pct" in top_rc

    # Measured latency
    assert data["latency_ms"] > 0.0
    assert "x-process-time-ms" in response.headers


# -----------------------------------------------------------------------------
# 3. Batch Scoring Tests & Empirical Latency Measurement
# -----------------------------------------------------------------------------
def test_predict_batch_transactions(client: TestClient, sample_transactions: list[dict[str, Any]]):
    """Verify batch scoring on multiple transactions."""
    batch_payload = {
        "transactions": [
            {
                "TransactionID": s.get("transaction_id", 3000000 + i),
                "TransactionDT": s.get("transaction_dt", 12200000),
                "TransactionAmt": float(s.get("transaction_amt", 50.0 + i * 10)),
                "ProductCD": s.get("product_cd", "W"),
                "card1": s.get("card1", 1000 + i),
                "card4": s.get("card4", "visa"),
                "card6": s.get("card6", "debit"),
                "P_emaildomain": s.get("p_emaildomain", "gmail.com"),
            }
            for i, s in enumerate(sample_transactions[:5])
        ],
        "top_k": 5,
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5
    assert data["average_latency_ms"] > 0.0


def test_latency_benchmark_and_reporting(client: TestClient, sample_transactions: list[dict[str, Any]]):
    """
    Measure actual empirical latency across repeated calls and calculate p50, p95, and p99.
    Demonstrates transparent measured latency rather than assuming fixed bounds.
    """
    sample = sample_transactions[0]
    payload = {
        "TransactionID": sample.get("transaction_id", 3459526),
        "TransactionDT": sample.get("transaction_dt", 12198174),
        "TransactionAmt": float(sample.get("transaction_amt", 117.0)),
        "ProductCD": sample.get("product_cd", "W"),
        "card1": sample.get("card1", 4436),
    }

    latencies = []
    # Warm-up call
    _ = client.post("/predict", json=payload)

    # 10 benchmark calls
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload)
        lat = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies.append(lat)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    print(f"\n[LATENCY BENCHMARK] Measured p50: {p50:.2f}ms | p95: {p95:.2f}ms | p99: {p99:.2f}ms")
    assert p50 > 0.0
    assert p95 >= p50


# -----------------------------------------------------------------------------
# 4. Security & Input Validation Tests
# -----------------------------------------------------------------------------
def test_validation_negative_amount_rejected(client: TestClient):
    """Verify physical bound enforcement: negative amount returns 422."""
    payload = {
        "TransactionID": 99999,
        "TransactionAmt": -50.0,
        "ProductCD": "W",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_payload_size_limit_guard(client: TestClient):
    """Verify oversized payload middleware rejection (> 1MB)."""
    # Create header claiming oversized payload
    headers = {"Content-Length": str(2 * 1024 * 1024)}
    response = client.post("/predict", json={"TransactionAmt": 100.0}, headers=headers)
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]


# -----------------------------------------------------------------------------
# 5. Held-Out Demo Replay Endpoint Tests
# -----------------------------------------------------------------------------
def test_replay_stream_pagination(client: TestClient):
    """Verify paginated stream of held-out test transactions."""
    response = client.get("/replay?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["transactions"]) == 10
    assert data["total_count"] == 1500
    assert data["limit"] == 10
    assert data["offset"] == 0

    first_tx = data["transactions"][0]
    assert "transaction_id" in first_tx
    assert "is_fraud" in first_tx
    assert "grounded_narrative" in first_tx


def test_replay_single_by_id(client: TestClient):
    """Verify lookup of a specific held-out transaction by ID."""
    response = client.get("/replay/3459526")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 3459526
    assert data["is_fraud"] in (0, 1)


def test_replay_not_found(client: TestClient):
    """Verify 404 on non-existent transaction ID."""
    response = client.get("/replay/999999999")
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# 6. Operational vs Evaluation Monitoring Separation Tests
# -----------------------------------------------------------------------------
def test_operational_monitoring_unlabeled(client: TestClient):
    """Verify unlabeled operational metrics endpoint."""
    response = client.get("/monitoring/operational")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions_logged" in data
    assert "risk_tier_distribution" in data
    assert "score_distribution_deciles" in data
    assert "high_risk_queue_depth" in data
    assert len(data["score_distribution_deciles"]) == 10


def test_evaluation_monitoring_labeled_benchmark(client: TestClient):
    """Verify labeled evaluation benchmark metrics endpoint."""
    response = client.get("/monitoring/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_description"] == "Benchmark on Held-Out Labeled Test Replay"
    assert data["total_evaluated"] == 1500
    assert "precision_at_high_tier" in data
    assert "recall_at_high_tier" in data
    assert "confusion_matrix" in data
    cm = data["confusion_matrix"]
    assert cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"] == 1500


def test_high_risk_review_queue(client: TestClient):
    """Verify prioritized high-risk review queue endpoint."""
    response = client.get("/monitoring/review-queue?limit=10")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    if items:
        # Check sorting by probability descending
        probs = [it["predicted_probability"] for it in items]
        assert probs == sorted(probs, reverse=True)


# -----------------------------------------------------------------------------
# 7. Resilient Database Fallback Tests
# -----------------------------------------------------------------------------
def test_database_graceful_degradation(client: TestClient):
    """Verify inference scoring continues seamlessly even if database connection fails."""
    # Temporarily set is_connected = False to simulate cloud outage
    original_state = db_manager.is_connected
    try:
        db_manager.is_connected = False
        payload = {
            "TransactionID": 888888,
            "TransactionAmt": 250.0,
            "ProductCD": "W",
            "card1": 5000,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_probability"] >= 0.0
    finally:
        db_manager.is_connected = original_state
