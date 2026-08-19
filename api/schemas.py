"""
api/schemas.py
--------------
Pydantic Request & Response schemas for FastAPI serving layer.
Strictly adheres to honest metric naming (predicted_probability) and enforces
physical bounds and input validation.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# 1. Explainability & Reason Code Schemas
# -----------------------------------------------------------------------------
class ReasonCodeItem(BaseModel):
    """Structured SHAP-derived reason code attribution item."""
    model_config = ConfigDict(extra="ignore")

    feature: str = Field(..., description="Original feature identifier")
    display_name: str = Field(..., description="Human-readable domain label")
    feature_value: Any = Field(..., description="Observed feature value in transaction")
    shap_value: float = Field(..., description="SHAP feature attribution in log-odds")
    contribution_pct: float = Field(..., description="Normalized contribution percentage")
    direction: Literal["INCREASES_RISK", "DECREASES_RISK", "REDUCES_RISK"] = Field(
        ..., description="Directional risk impact"
    )
    category: str = Field(..., description="Domain grouping category")
    description: str = Field(..., description="Plain-language description of feature meaning")
    is_collinear_cluster: bool = Field(
        False, description="True if feature consolidates multiple collinear attributes"
    )
    cluster_members: list[str] | None = Field(
        None, description="Member features consolidated into this reason code if collinear"
    )


# -----------------------------------------------------------------------------
# 2. Inference Input & Output Schemas
# -----------------------------------------------------------------------------
class TransactionInput(BaseModel):
    """
    Inbound raw transaction record payload.
    Supports either minimal required attributes or the full 400+ raw IEEE-CIS feature dictionary.
    """
    model_config = ConfigDict(extra="allow")

    TransactionID: int | None = Field(None, description="Unique transaction identifier")
    TransactionDT: int | None = Field(None, description="Relative timestamp delta in seconds")
    TransactionAmt: float = Field(..., gt=0.0, description="Transaction payment amount in USD (must be > 0)")
    ProductCD: str | None = Field("W", description="Product / transaction channel category code")
    card1: int | None = Field(None, description="Primary payment card identification proxy")
    card2: float | None = Field(None, description="Secondary card sub-classification code")
    card3: float | None = Field(None, description="Card issuer country/type code")
    card4: str | None = Field(None, description="Payment card network brand (visa, mastercard, discover, etc.)")
    card5: float | None = Field(None, description="Card bank sub-category code")
    card6: str | None = Field(None, description="Card funding type (credit, debit, charge card)")
    addr1: float | None = Field(None, description="Billing region / zip code proxy")
    addr2: float | None = Field(None, description="Billing country code proxy")
    P_emaildomain: str | None = Field(None, description="Purchaser email domain")
    R_emaildomain: str | None = Field(None, description="Recipient email domain")


class PredictionResponse(BaseModel):
    """
    Standardized single-transaction inference response.
    Includes predicted probability, risk tier, action, reason codes, and measured latency.
    """
    model_config = ConfigDict(extra="ignore")

    prediction_id: str = Field(..., description="Unique UUID for this prediction event")
    transaction_id: int | None = Field(None, description="Input transaction identifier")
    predicted_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model predicted fraud risk probability [0.0, 1.0]"
    )
    predicted_risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Operational risk classification tier"
    )
    decision_action: Literal["APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW"] = Field(
        ..., description="Recommended operational business action"
    )
    recommended_workflow: str = Field(
        ..., description="Actionable execution workflow instruction for operations team"
    )
    top_reason_codes: list[ReasonCodeItem] = Field(
        default_factory=list, description="Top positive risk-increasing feature attributions"
    )
    mitigating_factors: list[ReasonCodeItem] = Field(
        default_factory=list, description="Top negative risk-decreasing feature attributions"
    )
    grounded_narrative: str | None = Field(
        None, description="Validated template-grounded analyst narrative summary"
    )
    model_version: str = Field("v1.0.0", description="Model booster artifact version")
    latency_ms: float = Field(..., description="Actual measured inference & explanation latency in milliseconds")
    created_at: str = Field(..., description="ISO 8601 prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Batch scoring request payload."""
    transactions: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=10, description="Number of top reason codes to extract per transaction")


class BatchPredictionResponse(BaseModel):
    """Batch scoring response."""
    predictions: list[PredictionResponse]
    total_processed: int
    average_latency_ms: float


# -----------------------------------------------------------------------------
# 3. Demo Replay Stream Schemas
# -----------------------------------------------------------------------------
class ReplayTransactionItem(BaseModel):
    """Held-out test transaction replay item for the live interactive demo."""
    model_config = ConfigDict(extra="allow")

    transaction_id: int
    transaction_dt: int
    transaction_amt: float
    product_cd: str
    card1: int
    card4: str | None = None
    card6: str | None = None
    p_emaildomain: str | None = None
    is_fraud: int = Field(..., ge=0, le=1, description="Held-out ground truth label for simulated evaluation")
    fraud_probability: float | None = None
    predicted_risk_tier: str | None = None
    decision_action: str | None = None
    top_reason_codes: list[dict[str, Any]] | None = None
    grounded_narrative: str | None = None


class ReplayListResponse(BaseModel):
    """Paginated list of replay transactions for the simulated streaming demo."""
    transactions: list[ReplayTransactionItem]
    total_count: int
    limit: int
    offset: int


# -----------------------------------------------------------------------------
# 4. Observability & Monitoring Schemas
# -----------------------------------------------------------------------------
class OperationalMetricsResponse(BaseModel):
    """
    Unlabeled operational observability metrics.
    Reflects real-world inference traffic without requiring ground-truth labels.
    """
    total_predictions_logged: int
    risk_tier_distribution: dict[str, int]
    risk_tier_percentages: dict[str, float]
    average_predicted_probability: float
    score_distribution_deciles: list[dict[str, Any]]
    high_risk_queue_depth: int
    period_start: str | None = None
    period_end: str | None = None


class ConfusionMatrixStats(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class EvaluationMetricsResponse(BaseModel):
    """
    Evaluation metrics computed on labeled held-out test / replay transactions.
    Strictly distinguished from unlabeled operational monitoring.
    """
    dataset_description: str = "Benchmark on Held-Out Labeled Test Replay"
    total_evaluated: int
    actual_fraud_rate: float
    pr_auc: float
    roc_auc: float
    precision_at_high_tier: float
    recall_at_high_tier: float
    false_positive_rate_at_high_tier: float
    confusion_matrix: ConfusionMatrixStats


class HighRiskQueueItem(BaseModel):
    """High-priority manual review queue item."""
    prediction_id: str
    transaction_id: int
    transaction_dt: int
    transaction_amt: float
    predicted_probability: float
    top_risk_feature: str | None = None
    grounded_narrative: str | None = None
    created_at: str


# -----------------------------------------------------------------------------
# 5. Service Health Schema
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """System health check and component initialization status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    app_name: str
    version: str
    environment: str
    model_loaded: bool
    pipeline_loaded: bool
    explainer_loaded: bool
    database_connected: bool
    benchmark_inference_latency_ms: float | None = None
