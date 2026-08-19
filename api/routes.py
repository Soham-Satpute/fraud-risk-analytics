"""
api/routes.py
-------------
FastAPI endpoint routes for the Fraud Risk Analytics & Detection System.
Includes live simulated inference, TreeSHAP reason code extraction, held-out demo replay,
and distinct operational vs labeled monitoring observability endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, status

from api.config import settings
from api.db import db_manager
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ConfusionMatrixStats,
    EvaluationMetricsResponse,
    HealthResponse,
    HighRiskQueueItem,
    OperationalMetricsResponse,
    PredictionResponse,
    ReasonCodeItem,
    ReplayListResponse,
    ReplayTransactionItem,
    TransactionInput,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_api_key(x_api_key: str | None = Header(None)) -> None:
    """Optional security guard verifying API key header if configured in environment."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# -----------------------------------------------------------------------------
# 1. Health & Readiness Endpoint
# -----------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Component Status",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Returns system readiness status, verifying model artifacts and database connectivity.
    Performs a single warm-up inference to measure and report actual benchmark latency.
    """
    explainer = getattr(request.app.state, "explainer", None)
    model_loaded = explainer is not None and getattr(explainer, "model", None) is not None
    pipeline_loaded = explainer is not None and getattr(explainer, "pipeline", None) is not None
    explainer_loaded = explainer is not None and getattr(explainer, "explainer", None) is not None
    db_connected = db_manager.check_connection()

    benchmark_latency = None
    if explainer_loaded:
        try:
            t0 = time.perf_counter()
            _ = explainer.explain_transaction(
                {"TransactionID": 9999999, "TransactionAmt": 100.0, "ProductCD": "W", "card1": 5000}
            )
            benchmark_latency = round((time.perf_counter() - t0) * 1000.0, 2)
        except Exception as e:
            logger.warning("Warm-up benchmark inference check failed: %s", str(e))

    overall_status = "healthy" if (model_loaded and explainer_loaded) else "degraded"

    return HealthResponse(
        status=overall_status,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        model_loaded=model_loaded,
        pipeline_loaded=pipeline_loaded,
        explainer_loaded=explainer_loaded,
        database_connected=db_connected,
        benchmark_inference_latency_ms=benchmark_latency,
    )


# -----------------------------------------------------------------------------
# 2. Single Transaction Scoring Endpoint (/predict)
# -----------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score Single Transaction",
    tags=["Inference"],
)
async def predict_single_transaction(
    payload: TransactionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(None),
) -> PredictionResponse:
    """
    Scores a single transaction against the Champion LightGBM fraud model.
    Computes TreeSHAP feature attributions, maps to risk tiers and operational decision actions,
    measures actual execution latency, and logs audit record asynchronously to PostgreSQL.
    """
    _verify_api_key(x_api_key)

    explainer = getattr(request.app.state, "explainer", None)
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model explainer engine is not initialized.",
        )

    t0 = time.perf_counter()
    raw_dict = payload.model_dump(exclude_none=True)

    try:
        explanation = explainer.explain_transaction(raw_dict, top_k=5)
    except Exception as e:
        logger.error("Inference execution error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute inference explanation for the provided payload.",
        )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    pred_id = str(uuid.uuid4())
    created_at_str = datetime.now(timezone.utc).isoformat()

    # Format reason codes for response schema
    top_reasons = [
        ReasonCodeItem(
            feature=rc["feature"],
            display_name=rc["display_name"],
            feature_value=rc["feature_value"],
            shap_value=round(float(rc["shap_value"]), 4),
            contribution_pct=round(float(rc["contribution_pct"]), 2),
            direction=rc["direction"],
            category=rc["category"],
            description=rc["description"],
            is_collinear_cluster=rc.get("is_collinear_cluster", False),
            cluster_members=rc.get("cluster_members"),
        )
        for rc in explanation.top_risk_factors
    ]

    mitigating_reasons = [
        ReasonCodeItem(
            feature=rc["feature"],
            display_name=rc["display_name"],
            feature_value=rc["feature_value"],
            shap_value=round(float(rc["shap_value"]), 4),
            contribution_pct=round(float(rc["contribution_pct"]), 2),
            direction=rc["direction"],
            category=rc["category"],
            description=rc["description"],
            is_collinear_cluster=rc.get("is_collinear_cluster", False),
            cluster_members=rc.get("cluster_members"),
        )
        for rc in explanation.top_mitigating_factors
    ]

    response = PredictionResponse(
        prediction_id=pred_id,
        transaction_id=payload.TransactionID,
        predicted_probability=round(float(explanation.fraud_probability), 4),
        predicted_risk_tier=explanation.predicted_risk_tier,
        decision_action=explanation.decision_action,
        recommended_workflow=explanation.recommended_workflow,
        top_reason_codes=top_reasons,
        mitigating_factors=mitigating_reasons,
        grounded_narrative=getattr(explanation, "grounded_narrative", None),
        model_version=settings.APP_VERSION,
        latency_ms=latency_ms,
        created_at=created_at_str,
    )

    # Log prediction to Supabase / PostgreSQL in background
    db_record = {
        "prediction_id": pred_id,
        "transaction_id": payload.TransactionID or 0,
        "transaction_dt": payload.TransactionDT or 0,
        "transaction_amt": payload.TransactionAmt,
        "fraud_probability": response.predicted_probability,
        "predicted_risk_tier": response.predicted_risk_tier,
        "decision_action": response.decision_action,
        "actual_label": raw_dict.get("isFraud", raw_dict.get("is_fraud")),
        "top_reason_codes": [rc.model_dump() for rc in top_reasons],
        "grounded_narrative": response.grounded_narrative,
        "model_version": response.model_version,
        "latency_ms": latency_ms,
        "created_at": created_at_str,
    }
    background_tasks.add_task(db_manager.insert_prediction, db_record)

    return response


# -----------------------------------------------------------------------------
# 3. Batch Scoring Endpoint (/predict/batch)
# -----------------------------------------------------------------------------
@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score Batch of Transactions",
    tags=["Inference"],
)
async def predict_batch_transactions(
    payload: BatchPredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(None),
) -> BatchPredictionResponse:
    """
    Scores a batch of transactions (1–1,000 records) sequentially or vectorized.
    Returns structured predictions with individual reason codes and overall batch latency.
    """
    _verify_api_key(x_api_key)

    explainer = getattr(request.app.state, "explainer", None)
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model explainer engine is not initialized.",
        )

    t0 = time.perf_counter()
    results: list[PredictionResponse] = []
    db_records: list[dict[str, Any]] = []

    for tx in payload.transactions:
        t_single = time.perf_counter()
        try:
            explanation = explainer.explain_transaction(tx, top_k=payload.top_k)
            single_lat = round((time.perf_counter() - t_single) * 1000.0, 2)
            pred_id = str(uuid.uuid4())
            created_at_str = datetime.now(timezone.utc).isoformat()

            top_reasons = [
                ReasonCodeItem(
                    feature=rc["feature"],
                    display_name=rc["display_name"],
                    feature_value=rc["feature_value"],
                    shap_value=round(float(rc["shap_value"]), 4),
                    contribution_pct=round(float(rc["contribution_pct"]), 2),
                    direction=rc["direction"],
                    category=rc["category"],
                    description=rc["description"],
                    is_collinear_cluster=rc.get("is_collinear_cluster", False),
                    cluster_members=rc.get("cluster_members"),
                )
                for rc in explanation.top_risk_factors
            ]

            mitigating_reasons = [
                ReasonCodeItem(
                    feature=rc["feature"],
                    display_name=rc["display_name"],
                    feature_value=rc["feature_value"],
                    shap_value=round(float(rc["shap_value"]), 4),
                    contribution_pct=round(float(rc["contribution_pct"]), 2),
                    direction=rc["direction"],
                    category=rc["category"],
                    description=rc["description"],
                    is_collinear_cluster=rc.get("is_collinear_cluster", False),
                    cluster_members=rc.get("cluster_members"),
                )
                for rc in explanation.top_mitigating_factors
            ]

            tx_amt = float(tx.get("TransactionAmt", 100.0))
            prob = round(float(explanation.fraud_probability), 4)

            item = PredictionResponse(
                prediction_id=pred_id,
                transaction_id=tx.get("TransactionID"),
                predicted_probability=prob,
                predicted_risk_tier=explanation.predicted_risk_tier,
                decision_action=explanation.decision_action,
                recommended_workflow=explanation.recommended_workflow,
                top_reason_codes=top_reasons,
                mitigating_factors=mitigating_reasons,
                grounded_narrative=getattr(explanation, "grounded_narrative", None),
                model_version=settings.APP_VERSION,
                latency_ms=single_lat,
                created_at=created_at_str,
            )
            results.append(item)

            db_records.append({
                "prediction_id": pred_id,
                "transaction_id": tx.get("TransactionID", 0),
                "transaction_dt": tx.get("TransactionDT", 0),
                "transaction_amt": tx_amt,
                "fraud_probability": prob,
                "predicted_risk_tier": explanation.predicted_risk_tier,
                "decision_action": explanation.decision_action,
                "actual_label": tx.get("isFraud", tx.get("is_fraud")),
                "top_reason_codes": [rc.model_dump() for rc in top_reasons],
                "grounded_narrative": item.grounded_narrative,
                "model_version": item.model_version,
                "latency_ms": single_lat,
                "created_at": created_at_str,
            })
        except Exception as e:
            logger.warning("Error scoring record in batch: %s", str(e))

    total_batch_lat = round((time.perf_counter() - t0) * 1000.0, 2)
    avg_lat = round(total_batch_lat / max(len(results), 1), 2)

    # Bulk insert in background
    if db_records:
        background_tasks.add_task(db_manager.insert_predictions_batch, db_records)

    return BatchPredictionResponse(
        predictions=results,
        total_processed=len(results),
        average_latency_ms=avg_lat,
    )


# -----------------------------------------------------------------------------
# 4. Held-Out Demo Replay Endpoints (/replay)
# -----------------------------------------------------------------------------
@router.get(
    "/replay",
    response_model=ReplayListResponse,
    summary="Stream Held-Out Test Transactions",
    tags=["Demo Replay"],
)
async def get_demo_replay_stream(
    limit: int = Query(20, ge=1, le=100, description="Page batch size"),
    offset: int = Query(0, ge=0, description="Offset position in demo slice"),
) -> ReplayListResponse:
    """
    Returns a paginated stream of curated held-out test transactions (TransactionDT > 12,192,854)
    with pre-computed TreeSHAP reason codes and validated offline grounded narratives.
    """
    rows, total_count = db_manager.fetch_demo_replay(limit=limit, offset=offset)
    items = [
        ReplayTransactionItem(
            transaction_id=r.get("transaction_id", 0),
            transaction_dt=r.get("transaction_dt", 0),
            transaction_amt=float(r.get("transaction_amt", 0.0)),
            product_cd=r.get("product_cd", "W"),
            card1=r.get("card1", 0),
            card4=r.get("card4"),
            card6=r.get("card6"),
            p_emaildomain=r.get("p_emaildomain"),
            is_fraud=int(r.get("is_fraud", 0)),
            fraud_probability=r.get("fraud_probability"),
            predicted_risk_tier=r.get("predicted_risk_tier"),
            decision_action=r.get("decision_action"),
            top_reason_codes=r.get("top_reason_codes"),
            grounded_narrative=r.get("grounded_narrative"),
        )
        for r in rows
    ]

    return ReplayListResponse(
        transactions=items,
        total_count=total_count,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/replay/{transaction_id}",
    response_model=ReplayTransactionItem,
    summary="Get Single Held-Out Transaction by ID",
    tags=["Demo Replay"],
)
async def get_demo_transaction_by_id(transaction_id: int) -> ReplayTransactionItem:
    """Fetches a specific held-out transaction by its unique TransactionID."""
    row = db_manager.fetch_demo_replay_by_id(transaction_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Held-out transaction with TransactionID={transaction_id} not found.",
        )

    return ReplayTransactionItem(
        transaction_id=row.get("transaction_id", 0),
        transaction_dt=row.get("transaction_dt", 0),
        transaction_amt=float(row.get("transaction_amt", 0.0)),
        product_cd=row.get("product_cd", "W"),
        card1=row.get("card1", 0),
        card4=row.get("card4"),
        card6=row.get("card6"),
        p_emaildomain=row.get("p_emaildomain"),
        is_fraud=int(row.get("is_fraud", 0)),
        fraud_probability=row.get("fraud_probability"),
        predicted_risk_tier=row.get("predicted_risk_tier"),
        decision_action=row.get("decision_action"),
        top_reason_codes=row.get("top_reason_codes"),
        grounded_narrative=row.get("grounded_narrative"),
    )


# -----------------------------------------------------------------------------
# 5. Observability & Monitoring Endpoints
# -----------------------------------------------------------------------------
@router.get(
    "/monitoring/operational",
    response_model=OperationalMetricsResponse,
    summary="Unlabeled Operational Monitoring",
    tags=["Monitoring"],
)
async def get_operational_monitoring() -> OperationalMetricsResponse:
    """
    Returns real-world operational observability metrics that do NOT require ground-truth labels:
    overall volume, score distribution across deciles, risk tier breakdown, and high-risk review backlog.
    """
    metrics = db_manager.fetch_operational_metrics()
    return OperationalMetricsResponse(**metrics)


@router.get(
    "/monitoring/evaluation",
    response_model=EvaluationMetricsResponse,
    summary="Held-Out Labeled Test Performance Benchmark",
    tags=["Monitoring"],
)
async def get_labeled_evaluation_monitoring() -> EvaluationMetricsResponse:
    """
    Returns precision, recall, FPR, and confusion matrix computed strictly on held-out test data
    where ground-truth actual labels are known.
    """
    eval_metrics = db_manager.fetch_labeled_evaluation_metrics()
    return EvaluationMetricsResponse(**eval_metrics)


@router.get(
    "/monitoring/review-queue",
    response_model=list[HighRiskQueueItem],
    summary="High-Risk Manual Review Backlog",
    tags=["Monitoring"],
)
async def get_high_risk_review_queue(
    limit: int = Query(50, ge=1, le=200, description="Max high-risk queue items to return"),
) -> list[HighRiskQueueItem]:
    """
    Fetches prioritized high-risk transactions requiring manual fraud analyst review (predicted_risk_tier = 'HIGH').
    """
    queue_rows = db_manager.fetch_high_risk_queue(limit=limit)
    return [HighRiskQueueItem(**r) for r in queue_rows]
