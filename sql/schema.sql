-- ============================================================================
-- SQL Schema: Fraud Risk Analytics & Detection System
-- Architecture: PostgreSQL Serving & Logging Layer (Supabase Free Tier)
-- ============================================================================
-- IMPORTANT ARCHITECTURAL BOUNDARY:
-- This database serves strictly as an operational inference, audit, and demo
-- replay layer. Full raw or training data (590k rows × 434 cols) is NEVER loaded
-- into Postgres to respect Supabase free-tier storage caps (<500MB).
-- Raw and feature-engineered datasets remain stored locally as columnar Parquet.
-- ============================================================================

-- Enable pgcrypto for UUID generation if needed
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ----------------------------------------------------------------------------
-- 1. Table: predictions
-- Purpose: Real-time and batched inference audit log. Stores model outputs,
--          decision action tiers, SHAP-derived reason codes, and LLM narratives.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id BIGINT NOT NULL,
    transaction_dt INT NOT NULL,
    transaction_amt NUMERIC(10, 2) NOT NULL CHECK (transaction_amt > 0),
    fraud_probability REAL NOT NULL CHECK (fraud_probability >= 0.0 AND fraud_probability <= 1.0),
    predicted_risk_tier VARCHAR(20) NOT NULL CHECK (predicted_risk_tier IN ('LOW', 'MEDIUM', 'HIGH')),
    decision_action VARCHAR(30) NOT NULL CHECK (decision_action IN ('APPROVE', 'STEP_UP_AUTH', 'MANUAL_REVIEW')),
    actual_label SMALLINT NULL CHECK (actual_label IN (0, 1)), -- Nullable for unlabeled live inference
    top_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    grounded_narrative TEXT NULL,
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0.0',
    latency_ms REAL NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Comments on predictions table and columns
COMMENT ON TABLE predictions IS 'Operational audit log for model inference predictions and explainability payloads';
COMMENT ON COLUMN predictions.fraud_probability IS 'Calibrated or raw risk probability score output by the model [0.0, 1.0]';
COMMENT ON COLUMN predictions.predicted_risk_tier IS 'Discretized operational risk tier: LOW, MEDIUM, HIGH';
COMMENT ON COLUMN predictions.decision_action IS 'Business action triggered: APPROVE (low), STEP_UP_AUTH (medium), MANUAL_REVIEW (high)';
COMMENT ON COLUMN predictions.top_reason_codes IS 'Top-k aggregated SHAP feature attributions and directions';
COMMENT ON COLUMN predictions.grounded_narrative IS 'Template-constrained LLM narrative validated against SHAP evidence';

-- Optimized Indexes for predictions
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_tier ON predictions(predicted_risk_tier);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_prob ON predictions(fraud_probability DESC);

-- Partial index for high-risk / manual review queries (efficient review queue retrieval)
CREATE INDEX IF NOT EXISTS idx_predictions_manual_review_queue 
ON predictions(created_at DESC) 
WHERE predicted_risk_tier = 'HIGH';


-- ----------------------------------------------------------------------------
-- 2. Table: model_runs
-- Purpose: System of record for model versions, threshold policies, validation
--          metrics, and financial cost-benefit outcomes.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    run_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    validation_strategy VARCHAR(100) NOT NULL,
    operating_threshold REAL NOT NULL CHECK (operating_threshold >= 0.0 AND operating_threshold <= 1.0),
    total_evaluated INT NOT NULL CHECK (total_evaluated > 0),
    pr_auc REAL NULL,
    roc_auc REAL NULL,
    recall_at_1pct_fpr REAL NULL,
    recall_at_5pct_fpr REAL NULL,
    cost_saved_min NUMERIC(12, 2) NULL,
    cost_saved_max NUMERIC(12, 2) NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE model_runs IS 'Model registry and offline evaluation benchmark logs';
COMMENT ON COLUMN model_runs.validation_strategy IS 'Split methodology: e.g., Temporal Split (TransactionDT <= 12,192,854)';
COMMENT ON COLUMN model_runs.cost_saved_min IS 'Lower bound of estimated financial savings under stated cost assumptions';
COMMENT ON COLUMN model_runs.cost_saved_max IS 'Upper bound of estimated financial savings under stated cost assumptions';


-- ----------------------------------------------------------------------------
-- 3. Table: demo_replay
-- Purpose: Curated slice (~1,000–2,000 rows) of real held-out test transactions
--          used by the FastAPI / Next.js app to simulate a streaming feed.
-- Storage Footprint: < 2 MB (well within free tier limits).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS demo_replay (
    demo_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL UNIQUE,
    transaction_dt INT NOT NULL,
    transaction_amt NUMERIC(10, 2) NOT NULL,
    product_cd VARCHAR(10) NOT NULL,
    card1 INT NOT NULL,
    card4 VARCHAR(30) NULL,
    card6 VARCHAR(30) NULL,
    p_emaildomain VARCHAR(50) NULL,
    c1 REAL NULL,
    d1 REAL NULL,
    is_fraud SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    feature_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    grounded_narrative TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE demo_replay IS 'Curated held-out test transactions for simulated real-time inference in the portfolio demo';
COMMENT ON COLUMN demo_replay.feature_payload IS 'Full feature vector for passing directly to the model endpoint';

CREATE INDEX IF NOT EXISTS idx_demo_replay_tx_dt ON demo_replay(transaction_dt ASC);
CREATE INDEX IF NOT EXISTS idx_demo_replay_is_fraud ON demo_replay(is_fraud);
