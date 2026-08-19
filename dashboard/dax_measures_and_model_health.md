# Power BI DAX Measures & Model Health Specifications

> **Project:** Fraud Risk Analytics & Detection System  
> **Topic:** DAX Formula Reference, Cost-Saved Range Modeling, and Model Health Observability  
> **Source Plan:** `fintech-fraud-analytics-plan-v6.md` (§7, §11, Week 7) & `AGENTS.md` (§2, §4)

---

## 1. Operational DAX Measures (Unlabeled Monitoring)

These measures reflect real-time production traffic without requiring ground-truth fraud labels.

```dax
// 1. Total Volume Processed
Total_Transactions_Scored = COUNTROWS(predictions)

// 2. High-Risk Alert Volume (Manual Review Queue)
High_Risk_Volume = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] = "HIGH"
)

// 3. Operational Alert Rate (%)
High_Risk_Alert_Rate_Pct = 
DIVIDE([High_Risk_Volume], [Total_Transactions_Scored], 0) * 100

// 4. Medium-Risk Volume (Step-Up Authentication / 3DS)
Medium_Risk_Volume = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] = "MEDIUM"
)

// 5. Low-Risk Volume (Straight-Through Approvals)
Low_Risk_Volume = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] = "LOW"
)

// 6. Average Predicted Risk Score
Average_Predicted_Probability = AVERAGE(predictions[fraud_probability])
```

---

## 2. Labeled Benchmark DAX Measures (Held-Out Test Replay)

> **Important Standard:** These measures are strictly labeled in dashboards as *"Benchmark on Held-Out Labeled Replay"* to ensure transparency.

```dax
// 1. Ground Truth Fraud Count
Ground_Truth_Fraud_Count = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[actual_label] = 1
)

// 2. True Positives (High Risk Flagged AND Actual Fraud)
True_Positives = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] = "HIGH",
    predictions[actual_label] = 1
)

// 3. False Positives (High Risk Flagged BUT Legitimate)
False_Positives = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] = "HIGH",
    predictions[actual_label] = 0
)

// 4. False Negatives (Not Flagged High Risk BUT Actual Fraud)
False_Negatives = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] <> "HIGH",
    predictions[actual_label] = 1
)

// 5. True Negatives (Not Flagged High Risk AND Legitimate)
True_Negatives = 
CALCULATE(
    COUNTROWS(predictions),
    predictions[predicted_risk_tier] <> "HIGH",
    predictions[actual_label] = 0
)

// 6. Realized Precision (%)
Precision_Pct = 
DIVIDE([True_Positives], [True_Positives] + [False_Positives], 0) * 100

// 7. Realized Recall (%)
Recall_Pct = 
DIVIDE([True_Positives], [True_Positives] + [False_Negatives], 0) * 100

// 8. False Positive Rate (FPR %)
False_Positive_Rate_Pct = 
DIVIDE([False_Positives], [False_Positives] + [True_Negatives], 0) * 100
```

---

## 3. Parameterized Cost-Saved Range Model

Rather than presenting a single unqualified dollar figure, the cost savings model is parameterized using **dynamic Power BI scenario sliders** and reported as an explicit **range with stated assumptions**.

### Parameter Slices (What-If Parameters in Power BI)
- **Assumed Fraud Loss ($/tx):** Default `$200` (Slider Range: `$150` to `$250`)
- **Manual Review Cost ($/tx):** Default `$8` (Slider Range: `$5` to `$12`)

### DAX Implementation
```dax
// Estimated Gross Fraud Loss Prevented ($)
Gross_Fraud_Loss_Prevented = 
[True_Positives] * SELECTEDVALUE('Assumed Fraud Loss'[Loss_Per_Fraud], 200)

// Total Manual Investigation Operational Cost ($)
Total_Review_Cost = 
[High_Risk_Volume] * SELECTEDVALUE('Review Cost'[Cost_Per_Review], 8)

// Estimated Net Financial Savings ($)
Net_Cost_Saved_Point_Estimate = 
[Gross_Fraud_Loss_Prevented] - [Total_Review_Cost]

// Cost Saved Lower Bound ($) (Conservative: $150 loss caught, $12 review cost)
Cost_Saved_Lower_Bound = 
([True_Positives] * 150) - ([High_Risk_Volume] * 12)

// Cost Saved Upper Bound ($) (Optimistic: $250 loss caught, $5 review cost)
Cost_Saved_Upper_Bound = 
([True_Positives] * 250) - ([High_Risk_Volume] * 5)

// Formatted Range Display Label
Cost_Saved_Range_Display = 
"$" & FORMAT([Cost_Saved_Lower_Bound], "#,##0") & " – $" & FORMAT([Cost_Saved_Upper_Bound], "#,##0")
```

---

## 4. Model Health Observability Tile Specification

The **Model Health** tile provides a consolidated operational status for technical and business stakeholders:

| Tile Component | Metric / Calculation | Target / Baseline | Visual Type |
|---|---|---|---|
| **1. Ingestion Volume** | `[Total_Transactions_Scored]` | Operational throughput counter | Card |
| **2. High-Risk Backlog** | `[High_Risk_Volume]` (`tier = 'HIGH'`) | Daily analyst review capacity (<15%) | Card (Color threshold) |
| **3. Test Replay Precision** | `[Precision_Pct]` (Held-Out Replay) | Baseline: 15.08% / Champion: 46.6% | Gauge |
| **4. Test Replay Recall** | `[Recall_Pct]` (Held-Out Replay) | Champion Target: 65.95% | Gauge |
| **5. Decile Score Stability** | Histogram of `[fraud_probability]` in 10 deciles | Uniform low skewness; alert if Decile 10 > 25% | Clustered Column Chart |
| **6. Drift Alert Flag** | `IF([High_Risk_Alert_Rate_Pct] > 25%, "⚠️ ELEVATED DRIFT ALERT", "🟢 NORMAL STABILITY")` | Real-time threshold monitoring | Status Banner Card |
