-- ============================================================================
-- SQL Analytics Queries: Power BI & Operational Monitoring
-- System: Fraud Risk Analytics & Detection System
-- Database: PostgreSQL (Supabase Free Tier)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Daily Volume, Fraud Rate, and Risk Tier Breakdown
-- Used in Power BI: Overview tile & Daily Trends chart
-- ----------------------------------------------------------------------------
SELECT 
    DATE_TRUNC('day', created_at) AS date_bucket,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE predicted_risk_tier = 'HIGH') AS high_risk_count,
    COUNT(*) FILTER (WHERE predicted_risk_tier = 'MEDIUM') AS medium_risk_count,
    COUNT(*) FILTER (WHERE predicted_risk_tier = 'LOW') AS low_risk_count,
    ROUND((COUNT(*) FILTER (WHERE predicted_risk_tier = 'HIGH')::NUMERIC / NULLIF(COUNT(*), 0)) * 100, 2) AS high_risk_pct,
    ROUND(AVG(fraud_probability)::NUMERIC, 4) AS avg_fraud_probability,
    ROUND(AVG(transaction_amt)::NUMERIC, 2) AS avg_transaction_amount
FROM predictions
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date_bucket DESC;


-- ----------------------------------------------------------------------------
-- 2. Empirical Performance Metrics (Precision, Recall, FPR) on Labeled Replay
-- Used in Power BI: Model Health & Performance Verification tile
-- Note: Evaluates rows where actual_label IS NOT NULL.
-- ----------------------------------------------------------------------------
WITH classification_counts AS (
    SELECT
        COUNT(*) AS total_labeled,
        COUNT(*) FILTER (WHERE actual_label = 1 AND predicted_risk_tier = 'HIGH') AS true_positives,
        COUNT(*) FILTER (WHERE actual_label = 0 AND predicted_risk_tier = 'HIGH') AS false_positives,
        COUNT(*) FILTER (WHERE actual_label = 1 AND predicted_risk_tier != 'HIGH') AS false_negatives,
        COUNT(*) FILTER (WHERE actual_label = 0 AND predicted_risk_tier != 'HIGH') AS true_negatives,
        COUNT(*) FILTER (WHERE actual_label = 1) AS total_actual_fraud,
        COUNT(*) FILTER (WHERE actual_label = 0) AS total_actual_legit
    FROM predictions
    WHERE actual_label IS NOT NULL
)
SELECT
    total_labeled,
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
    total_actual_fraud,
    total_actual_legit,
    -- Recall (Fraud Capture Rate) = TP / (TP + FN)
    ROUND((true_positives::NUMERIC / NULLIF(total_actual_fraud, 0)) * 100, 2) AS recall_pct,
    -- Precision = TP / (TP + FP)
    ROUND((true_positives::NUMERIC / NULLIF(true_positives + false_positives, 0)) * 100, 2) AS precision_pct,
    -- False Positive Rate = FP / (FP + TN)
    ROUND((false_positives::NUMERIC / NULLIF(total_actual_legit, 0)) * 100, 2) AS fpr_pct,
    -- Manual Review Rate = (TP + FP) / Total
    ROUND(((true_positives + false_positives)::NUMERIC / NULLIF(total_labeled, 0)) * 100, 2) AS review_rate_pct
FROM classification_counts;


-- ----------------------------------------------------------------------------
-- 3. High-Risk Manual Review Backlog Queue
-- Used in API / Operational Analyst Dashboard: Prioritized investigation queue
-- ----------------------------------------------------------------------------
SELECT 
    prediction_id,
    transaction_id,
    transaction_dt,
    transaction_amt,
    ROUND(fraud_probability::NUMERIC, 4) AS fraud_probability,
    decision_action,
    top_reason_codes,
    grounded_narrative,
    created_at
FROM predictions
WHERE predicted_risk_tier = 'HIGH'
ORDER BY fraud_probability DESC, created_at DESC
LIMIT 50;


-- ----------------------------------------------------------------------------
-- 4. Score Distribution & Drift Monitoring (Decile Buckets)
-- Used in Power BI: Observability & Score Distribution Stability
-- ----------------------------------------------------------------------------
SELECT
    WIDTH_BUCKET(fraud_probability, 0.0, 1.0, 10) AS score_bucket,
    CONCAT(ROUND((WIDTH_BUCKET(fraud_probability, 0.0, 1.0, 10) - 1) * 0.1, 1), ' - ', ROUND(WIDTH_BUCKET(fraud_probability, 0.0, 1.0, 10) * 0.1, 1)) AS score_range,
    COUNT(*) AS transaction_count,
    ROUND((COUNT(*)::NUMERIC / NULLIF(SUM(COUNT(*)) OVER (), 0)) * 100, 2) AS pct_of_total,
    ROUND(AVG(transaction_amt)::NUMERIC, 2) AS avg_amount,
    COUNT(*) FILTER (WHERE actual_label = 1) AS actual_fraud_count
FROM predictions
GROUP BY WIDTH_BUCKET(fraud_probability, 0.0, 1.0, 10)
ORDER BY score_bucket ASC;


-- ----------------------------------------------------------------------------
-- 5. Top SHAP Reason Code Frequency
-- Analyzes which features most frequently drive high-risk classifications
-- ----------------------------------------------------------------------------
SELECT 
    reason->>'feature' AS feature_name,
    reason->>'direction' AS direction,
    COUNT(*) AS trigger_count,
    ROUND(AVG((reason->>'importance')::NUMERIC), 4) AS avg_importance
FROM predictions,
LATERAL jsonb_array_elements(top_reason_codes) AS reason
WHERE predicted_risk_tier = 'HIGH'
GROUP BY reason->>'feature', reason->>'direction'
ORDER BY trigger_count DESC
LIMIT 10;
