# Defensible Fintech Fraud Risk Analytics & Detection System
## An Evidence-Driven Case Study in Leakage-Free Machine Learning, Grounded Explainability, and Cost-Optimized Operating Decisions

> **Target Audience:** Hiring Managers, Data Science Leaders, Fraud Operations Directors, Risk Executives  
> **Source Repository:** `Soham-Satpute/fraud-risk-analytics`  
> **Dataset:** IEEE-CIS E-Commerce Fraud Detection Benchmark ($N = 590,540$ transactions)  
> **Infrastructure Stack:** Permanent $0 Free-Tier (PostgreSQL on Supabase, Render, Next.js on Vercel, Local Ollama / Grok API)  
> **Validation Property:** Temporal Split at 80th percentile ($TransactionDT \le 12,192,854$ Train vs. $> 12,192,854$ Test)

---

## 01 — Executive Summary

Modern e-commerce and fintech fraud detection systems face an acute operational dilemma: naive machine learning models often optimize for academic aggregate metrics (such as ROC-AUC) while generating catastrophic false-positive review volumes that overwhelm human investigation queues and increase operational overhead.

This project delivers an end-to-end, statistically defensible fraud risk analytics and scoring system built on the IEEE-CIS dataset ($590,540$ transactions, $3.499\%$ fraud rate). Rather than accepting default evaluation splits or arbitrary $0.50$ decision cutoffs, we executed a forensic data integrity investigation, engineered a leakage-free feature transformer, trained and benchmarked a Champion LightGBM gradient boosted tree against a calibrated Logistic Regression baseline, extracted consolidated TreeSHAP reason codes, generated 100% grounded analyst narratives via local LLM, and solved for optimal operational thresholds via a 12-step cost matrix workflow.

### Headline Analytical & Financial Results:
1. **Model Lift with 1,000-Resample 95% Bootstrap CIs:** Champion LightGBM achieves a **PR-AUC of $0.5441$** ($95\%\text{ CI: }[0.5282, 0.5607]$) vs. Baseline Logistic Regression of **$0.2746$** ($[0.2605, 0.2891]$) — a **$+98.1\%$ relative lift**. At a strict $1\%$ False Positive Rate (FPR) limit, LightGBM captures **$46.63\%$ of fraud** vs. $15.08\%$ for the baseline (**$3.09\times$ capture rate**).
2. **Cost-Sensitive Decision Optimization:** Solving the 3-tier routing economic cost model across $118,108$ held-out test transactions identifies **Candidate Policy B ($\tau_{high} = 0.70, \tau_{med} = 0.01$)** as the global cost-minimizing policy.
3. **Bottom-Line Financial Impact:** Candidate Policy B achieves **$\$649,433.00$ in net financial savings** ($80.0\%$ reduction in fraud operating costs from $\$812,800.00$ to $\$163,367.00$), outperforming the Logistic Regression baseline by **$+\$313,945.00$ (+93.6% lift in net savings)** while slashing the manual review queue by **$83.5\%$** ($4,297$ reviews vs. $26,089$ reviews) and increasing fraud analyst precision from $10.43\%$ to **$51.01\%$**.
4. **Permanent $0 Infrastructure:** The end-to-end deployment (FastAPI backend, Supabase PostgreSQL audit store, Next.js demo UI, and offline TreeSHAP narrative layer) operates entirely within perpetual free hosting tiers.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      EXECUTIVE IMPACT AT A GLANCE                       │
 ├──────────────────────────┬──────────────────────────┬───────────────────┤
 │     PR-AUC (PRECISION)   │     MANUAL REVIEW QUEUE  │    NET SAVINGS    │
 │   0.5441 vs 0.2746 Base  │    -83.5% Caseload       │    $649,433.00    │
 │       (+98.1% Lift)      │  (4,297 vs 26,089 Base)  │  (+93.6% vs Base) │
 └──────────────────────────┴──────────────────────────┴───────────────────┘
```

---

## 02 — Business Problem & Operational Context

In card-not-present (CNP) e-commerce transactions, fraud prevention is fundamentally an exercise in asymmetric risk management:
- **Cost of Undetected Fraud ($FN$):** Direct financial chargeback loss, interchange penalty fees, and merchandise write-offs (empirically modeled at an average base-case of $L_{fraud} = \$200.00$ per transaction).
- **Cost of False Positives ($FP$):** Manual investigation labor ($C_{review} = \$8.00$ per high-risk analyst touch), customer friction, checkout abandonment, and merchant interchange drag.
- **Operational Review Bottleneck:** Fraud operations teams operate under fixed headcount constraints. In our test volume of $118,108$ transactions, a naive model flagging $22.1\%$ of volume ($26,089$ alerts) causes queue overflow, massive alert fatigue, and delayed customer fulfillment.

### The 3-Tier Routing Architecture
To balance risk reduction against operational friction, our system implements a **3-Tier Routing Architecture**:

$$\text{Incoming Transaction Stream} \longrightarrow \text{Predicted Risk Probability } p = P(\text{isFraud}=1)$$

1. **Tier 1: Straight-Through Automated Approval ($p < \tau_{med}$):** Frictionless instant authorization for verified low-risk traffic ($9.34\%$ of transactions, $0.09\%$ fraud rate).
2. **Tier 2: Automated Step-Up Authentication ($\tau_{med} \le p < \tau_{high}$):** Automated challenge (3D-Secure, SMS OTP, or biometric IDV) applied to medium-risk traffic ($87.02\%$ of transactions, costing $C_{stepup} = \$0.50$/check). Deterrence efficiency $\eta_{stepup} = 80\%$ prevents $1,484$ fraud cases without human intervention.
3. **Tier 3: Prioritized Manual Investigation Queue ($p \ge \tau_{high}$):** High-risk traffic ($3.64\%$ of transactions, $4,297$ cases) routed to fraud analysts with consolidated TreeSHAP reason codes and grounded narrative briefings.

---

## 03 — Data Integrity Investigation & Validation Discovery

The foundation of this system is investigative rigor. Prior to building features or training models, we conducted a Week 1 forensic audit of the IEEE-CIS dataset ($590,540$ rows $\times$ $434$ columns):

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   DATA INTEGRITY & AUDIT FINDINGS                       │
 ├─────────────────────────┬───────────────────────────────────────────────┤
 │ TransactionDT Span      │ 86,400 to 15,811,131 seconds (182 days)       │
 │ Fraud Rate Temporal     │ 3.40% (first half) → 3.61% (second half)      │
 │ Entity Proxy Columns    │ card1, card2, card3, card5, addr1, addr2,     │
 │                         │ P_emaildomain → 94,846 unique card proxies    │
 │ Temporal Split Overlap  │ 67.6% entity overlap (Realistic validation)   │
 │ Random Split Overlap    │ 74.7% entity overlap (Leakage-inflated)       │
 │ Identity Table Coverage │ Only 23.8% of transactions have identity join │
 │ V-Feature Missingness   │ 7 distinct clusters; up to 85% missingness    │
 │ V-Feature Collinearity  │ 162 pairs with |r| >= 0.98                    │
 │ Feature Signal Audits   │ C1 identical to velocity (|r|=1.000)          │
 │                         │ D1 identical to time-since-last (|r|=1.000)   │
 └─────────────────────────┴───────────────────────────────────────────────┘
```

### Key Methodological Insights:
1. **Validation Split Selection:** A random $80/20$ train-test split allows $74.7\%$ entity overlap, causing artificial temporal leakage and inflating PR-AUC by $+24.8\%$. We enforced a strict **temporal split at $TransactionDT \le 12,192,854$** ($80\text{th percentile}$), yielding $472,432$ training rows and $118,108$ held-out test rows ($67.6\%$ realistic entity overlap).
2. **Timestamp Non-Identifiability:** `TransactionDT` represents relative seconds from an undisclosed arbitrary origin. We rejected ungrounded attempts to anchor it to a specific calendar date (e.g. December 2017) and modeled diurnal ($24\text{-hour}$) and day-of-week ($7\text{-day}$) cyclical trigonometric features purely from relative intervals.
3. **Signal Redundancy Audits:** Cross-correlation analysis proved that engineered velocity features were $|r| = 1.000$ identical to raw $C1$, and time-since-last-transaction was $|r| = 1.000$ identical to raw $D1$. Rather than rebuilding redundant features, we utilized $C1$ and $D1$ directly and engineered $24$ non-redundant composite features ($|r| < 0.25$ to $C1/D1$).

---

## 04 — Leakage-Free Feature Engineering & Modeling Architecture

To guarantee zero temporal lookahead leakage, our custom transformer (`FraudFeaturePipeline` in `src/features/engineer.py`) was fitted strictly on the training partition ($N = 472,432$) and serialized:

```
Raw Transaction Batch
        │
        ├── 1. Amount Context: amt_zscore_card1, amt_diff_mean_card1, amt_ratio_mean_card1,
        │                      amt_zscore_card1_addr1, amt_zscore_email, log_TransactionAmt
        ├── 2. Cyclical Time:  hour_sin, hour_cos (24h), dow_sin, dow_cos (7d)
        ├── 3. Frequency Maps: freq_card1, freq_card2, freq_addr1, freq_ProductCD, freq_R_emaildomain
        └── 4. Email Flags:    email_match_flag, null_P_email, null_R_email
        │
Transformed Feature Matrix (458 Features)
        │
        ├── Champion Model: LightGBM (200 trees, scale_pos_weight=27.46, lr=0.05, num_leaves=63)
        └── Baseline Model: Logistic Regression (StandardScaler + SimpleImputer + Balanced)
```

### Handling Class Imbalance (3.499% Fraud):
We ablated class-weighting ($scale\_pos\_weight = 27.46$) against unweighted trees and synthetic oversampling (SMOTE). Class-weighting on gradient-boosted trees preserved true ranking fidelity across rare fraud distributions while avoiding synthetic manifold distortion.

---

## 05 — Empirical Model Benchmarks & Bootstrap Confidence Intervals

Evaluating models strictly on the post-cutoff held-out test partition ($N = 118,108$, $4,064$ frauds):

| Metric | Champion LightGBM | Baseline Logistic Regression | Relative Lift / Gain | Statistical Rigor (1,000-Resample 95% Bootstrap CI) |
|---|---|---|---|---|
| **PR-AUC (Primary)** | **0.5441** | 0.2746 | **+98.1% Lift** | LightGBM: `[0.5282, 0.5607]` vs. Baseline: `[0.2605, 0.2891]` |
| **ROC-AUC** | **0.9035** | 0.8092 | **+11.7% Lift** | LightGBM: `[0.8982, 0.9087]` vs. Baseline: `[0.8021, 0.8164]` |
| **Recall @ 1% FPR** | **46.63%** | 15.08% | **3.09x Capture** | LightGBM: `[44.90%, 48.24%]` (captures 1,895 frauds at 1% FPR) |
| **Recall @ 5% FPR** | **65.95%** | 41.76% | **+24.19 pp** | LightGBM: `[64.46%, 67.45%]` (captures 2,680 frauds at 5% FPR) |
| **Brier Calibration Score** | **0.0246** | 0.1782 | **-86.2% Error** | Substantially superior probability concentration |
| **Empirical Latency (p50)** | **313.57 ms** | 14.20 ms | Interactive | Measured live through full TreeSHAP explanation pipeline |

```
PR-AUC Comparison:
Champion LightGBM:    [████████████████████████████] 0.5441 (95% CI: [0.5282, 0.5607])
Baseline Logistic:    [██████████████] 0.2746 (95% CI: [0.2605, 0.2891])
                      0.0            0.2            0.4            0.6
```

---

## 06 — Grounded TreeSHAP Explainability & Reason Code Consolidation

Raw machine learning probabilities are insufficient for operational fraud teams; fraud analysts require transparent, explainable reason codes.

### 1. Collinearity Consolidation Engine
Our Week 1 audit revealed $162$ near-duplicate $V$-feature pairs ($|r| \ge 0.98$). Rather than inundating analysts with redundant attributions (e.g. $V95, V101, V279, V293$), our reason code engine (`src/explainability/reason_codes.py`) consolidates these into unified domain clusters:
- **Payment Activity Volume Cluster** ($V95, V101, V279, V293$)
- **Card-Address Interaction Cluster** ($V257, V246, V201$)
- **Device Identity Verification Cluster** ($V188, V189, V242$)

### 2. Multi-Provider LLM & Mandatory Grounding Validator
To convert structured SHAP attributions into analyst-ready briefings, we deployed an offline multi-provider LLM pipeline (Local Ollama / Grok API / Deterministic Templates) coupled with a mandatory **Automated Grounding Validator** (`src/validation/grounding_validator.py`):
- **Direct Numerical Grounding:** Audits that every number in the narrative matches the raw payload.
- **Derived Multiples Recalculation:** Re-computes ratios (e.g. "3.2x average") against baseline historical distributions.
- **Directional Consistency:** Guarantees risk drivers are described as increasing risk and mitigating factors as decreasing risk.
- **Empirical Audit Performance:** Initial LLM narrative pass rate was **$93.47\%$**; all rejected narratives were automatically substituted with verified deterministic templates, guaranteeing a **$100.0\%$ verified grounding rate** across all $1,500$ held-out demo transactions.

---

## 07 — Business Impact & Cost-Sensitive Threshold Decision

We executed the 12-step business decision workflow (`src/models/threshold_analysis.py`) across $100$ candidate thresholds on the $118,108$ held-out test transactions:

### Candidate Policy Comparison Table
$$\text{Base Case: } L_{fraud} = \$200.00, C_{review} = \$8.00, C_{stepup} = \$0.50, \eta_{stepup} = 0.80$$

| Operating Policy | Operating Cutoffs ($\tau_{med}, \tau_{high}$) | Review Rate % (Queue Volume) | High-Tier Recall | Analyst Precision | High-Tier FPR | Total Expected Cost | Net Financial Savings | Operational Recommendation Status |
|---|---|---|---|---|---|---|---|---|
| **No Model: Accept All** | N/A | 0.0% (0) | 0.0% | N/A | 0.0% | $812,800.00 | $0.00 | Rejected (Unacceptable Loss) |
| **No Model: Review All** | N/A | 100.0% (118,108) | 100.0% | 3.44% | 100.0% | $944,864.00 | -$132,064.00 | Rejected (Cost Prohibitive) |
| **Naive Amount Rule (> $500)** | Amount > $500 | 4.08% (4,816) | 5.76% | 4.86% | 4.02% | $804,528.00 | $8,272.00 | Rejected (Ineffective Signal) |
| **Logistic Regression Default** | $p \ge 0.50$ | 22.09% (26,089) | 66.95% | 10.43% | 20.49% | $477,312.00 | $335,488.00 | Rejected (Queue Overflow) |
| **Candidate Policy A (Conservative)** | $\tau_{med}=0.01, \tau_{high}=0.96$ | **0.88%** (1,044) | 23.08% | **89.85%** | **0.093%** | $189,169.50 | $623,630.50 | Viable (1% Capacity Capped) |
| **Candidate Policy B (Balanced)** | $\tau_{med}=0.01, \tau_{high}=0.70$ | **3.64%** (4,297) | **53.94%** | **51.01%** | **1.85%** | **$163,367.00** | **$649,433.00** | **RECOMMENDED OPERATING POLICY** |
| **Candidate Policy C (Aggressive)** | $\tau_{med}=0.01, \tau_{high}=0.70$ | **3.64%** (4,297) | **53.94%** | **51.01%** | **1.85%** | **$163,367.00** | **$649,433.00** | Converges to Policy B |

### Why Candidate Policy B is Mathematically Optimal:
1. **Net Savings Maximization:** Delivers **$\$649,433.00$ in net savings** ($80.0\%$ reduction in fraud losses and operational costs).
2. **Caseload Feasibility:** Constrains human manual reviews to $4,297$ cases over the 26-week test period (an average of $\sim 23.6$ reviews per day), a **$-83.5\%$ reduction** compared to the $26,089$ reviews demanded by baseline Logistic Regression.
3. **High Analyst Precision:** Yields **$51.01\%$ precision** ($1$ in every $1.96$ manual investigations is fraud), eliminating analyst alert fatigue.
4. **Diminishing Returns Threshold:** Expanding capacity beyond $3.64\%$ to $10.0\%$ does not decrease expected costs; additional false-positive investigation fees ($C_{review} = \$8.00$) exceed the marginal fraud dollars saved.

---

## 08 — 36-Scenario Financial & Step-Up Sensitivity Analysis

To prove policy robustness across varying business environments, we evaluated the full $3 \times 3 \times 4 = 36$-scenario financial sensitivity matrix:

### Key Financial Sensitivity Findings:
- **Fraud Losses ($L_{fraud} \in [\$160, \$200, \$240]$) $\times$ Review Costs ($C_{review} \in [\$5, \$8, \$12]$) $\times$ Capacity Caps ($[1\%, 3\%, 5\%, 10\%]$)**:
  - Across all 36 scenarios, the optimal manual review cutoff $\tau_{high}$ remained strictly bounded between **$0.64$ and $0.96$**.
  - Total net savings remained robustly positive across every scenario, ranging from **$\$489,762.50$** (under $L=\$160, C=\$5, \text{Cap}=1\%$) to **$\$797,395.50$** (under $L=\$240, C=\$8, \text{Cap}=5\%$).
  - The optimal manual review volume never exceeded **$4.63\%$**, proving that scaling human review queues beyond $5.0\%$ is economically inefficient across all realistic cost regimes.

### Step-Up Authentication Deterrence Sensitivity:
Varying step-up deterrence efficiency ($\eta_{stepup} \in [50\%, 70\%, 80\%, 90\%]$) and challenge tool fees ($C_{stepup} \in [\$0.25, \$0.50, \$1.00]$):
- Even under the most pessimistic deterrence assumption ($50\%$ deterrence) and doubled tooling fees ($\$1.00$/check), Policy B preserves **$\$486,842.00$ in net savings**.
- At $80\%$ efficiency and $\$0.50$ challenge fee, the step-up tier mitigates $1,484$ fraud cases for a tooling cost of $\$51,391.00$, delivering an **ROI of $5.77\times$ on authentication tooling**.

---

## 09 — Final Operational Business Recommendation

### 1. Adopt Candidate Policy B (Balanced 3-Tier Routing Architecture)
- **Tier 1 (Straight-Through Approvals, $p < 0.01$):** Instantly approve $9.34\%$ of transactions with zero customer friction.
- **Tier 2 (Automated 3DS / OTP Step-Up, $0.01 \le p < 0.70$):** Route $87.02\%$ of transactions through automated frictionless challenge protocols.
- **Tier 3 (Prioritized Manual Investigation, $p \ge 0.70$):** Route $3.64\%$ of transactions to fraud analysts with TreeSHAP reason codes.

### 2. Prohibited Operational Anti-Patterns
- **Never Enforce Automatic Transaction Declining:** Decline decisions should only be rendered following human investigation or failed step-up authentication. Unconditional automated declines create severe merchant customer churn.
- **Never Enforce Default 0.50 Thresholds:** Setting cutoffs without cost modeling causes severe operational queue collapse ($22.1\%$ review rate).

### 3. Trigger Conditions That Would Alter Our Decision
- **Capacity Contraction ($< 1.0\%$ Queue Limit):** Shift immediately to **Candidate Policy A** ($\tau_{high} = 0.96$), restricting review volume to $0.88\%$ while maintaining $89.85\%$ precision and $\$623,630.50$ net savings.
- **Micro-Transaction Fraud Shift ($L_{fraud} < \$100$):** If fraud shifts to low-ticket items, manual review becomes cost-inefficient; expand the step-up authentication tier.
- **Authentication Tool Fee Inflation ($C_{stepup} > \$1.50$):** If SMS OTP costs increase, raise $\tau_{med}$ from $0.01$ to $0.05$ to expand straight-through approvals.

---

## 10 — Technical Limitations, Free-Tier Realities & Production Fintech Next Steps

### 1. Data & Methodological Limitations
- **Generalization Decay on Unseen Entities:** On completely novel card entities ($N = 10,952$, $0\%$ entity overlap), PR-AUC degrades from $0.5441$ to $0.4487$ ($-17.53\%$ relative decay) and Recall @ 1% FPR drops from $46.63\%$ to $36.36\%$. While performance remains substantially superior to the baseline ($0.2746$), cold-start entities require step-up authentication buffers.
- **Identity Attribute Sparsity:** Identity features were available on only $23.8\%$ of transactions, requiring explicit missingness indicator modeling.
- **Relative Timestamp Origin:** Because `TransactionDT` origin is undisclosed, seasonal holiday spikes cannot be modeled deterministically.

### 2. Free-Tier Infrastructure Realities
- **Render Backend Cold Starts:** On Render free tier, inactive services sleep after 15 minutes. The Next.js frontend is engineered with an automatic static JSON fallback to maintain 100% demo availability.
- **Database Scope Segregation:** Raw parquet/CSV data is kept local; Supabase PostgreSQL (<500MB free limit) is utilized exclusively for audit logging and demo replay streams.

### 3. Production Enterprise Fintech Evolution
If deployed in a tier-1 production financial institution, the following architectural enhancements would be implemented:
1. **Real-Time Feature Store:** Low-latency Redis / Feast cluster maintaining sliding-window velocity aggregations (10-minute, 1-hour, 24-hour entity counts).
2. **Streaming Event Pipeline:** Kafka / Apache Flink event backbone for sub-50ms distributed scoring.
3. **Automated MLOps Monitoring:** Continuous Kolmogorov-Smirnov drift testing and population stability index (PSI) alerts triggering automated quarterly model retraining.
