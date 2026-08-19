# Fraud Risk Analytics & Detection System
### With Grounded GenAI Analyst Explanations as a Supporting Capability

[![CI Pipeline](https://github.com/Soham-Satpute/fraud-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Soham-Satpute/fraud-risk-analytics/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-336791.svg)](https://supabase.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-72%2F72%20Passing-success.svg)](./tests)
[![Infrastructure](https://img.shields.io/badge/Infrastructure-%240%20Permanent%20Free%20Tier-orange.svg)](#infrastructure-architecture--permanent-0-stack)

> **Portfolio Demo:** Deployment-ready portfolio demo illustrating an end-to-end fintech fraud risk analytics and modeling system. Evaluates $590,540$ IEEE-CIS e-commerce transactions with temporal validation, TreeSHAP reason codes, 100% grounded local GenAI analyst summaries, lightweight FastAPI serving, and an empirical 3-tier operating cost optimization.

---

## 1. System Architecture & Data Flow

The system segregates heavy machine learning pipelines from high-throughput serving and analytical reporting layers:

```
                                      DATA & TRAINING PIPELINE
  ┌───────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
  │ IEEE-CIS Raw Dataset  │ ────> │ Leakage-Free Pipeline  │ ────> │ Champion LightGBM Tree │
  │ 590,540 Rows (Parquet)│       │ 24 Features (Train Fit)│       │ scale_pos_weight=27.46 │
  └───────────────────────┘       └────────────────────────┘       └───────────┬────────────┘
                                                                               │
                                                                               ▼
                                                                   ┌────────────────────────┐
                                                                   │ TreeSHAP Explainer     │
                                                                   │ 162 Collinear V-Pairs  │
                                                                   │ Consolidated to Groups │
                                                                   └───────────┬────────────┘
                                                                               │
                                                                               ▼
                                                                   ┌────────────────────────┐
                                                                   │ Grounding Validator    │
                                                                   │ Offline GenAI Briefing │
                                                                   │ (100% Grounded Audited)│
                                                                   └───────────┬────────────┘
                                                                               │
                                    SERVING & REPORTING LAYER                  │
                                                                               ▼
  ┌───────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
  │ Next.js Frontend Demo │ <───> │ FastAPI Backend Server │ <───> │ Supabase PostgreSQL    │
  │ 2 Pages (Vercel)      │       │ /predict & /replay     │       │ Audit Logs & Replay    │
  │ • Interactive Demo    │       │ Measured p50: ~313ms   │       │ predictions table      │
  │ • Methodology Summary │       │ Render Free Tier       │       │ <500MB Free Tier       │
  └───────────────────────┘       └────────────────────────┘       └───────────┬────────────┘
                                                                               │
                                                                               ▼
                                                                   ┌────────────────────────┐
                                                                   │ Power BI Analytics     │
                                                                   │ Parameterized DAX Cost │
                                                                   │ Model & Model Health   │
                                                                   └────────────────────────┘
```

---

## 2. Core Philosophy & Architectural Boundaries

1. **Investigation & Modeling are the Product:** The core differentiator is investigative rigor (entity overlap, temporal leakage, $V/D/C$ feature audits, class imbalance handling, cost-sensitive threshold optimization). GenAI is only a supporting explanation layer, not the headline.
2. **Honest & Defensible Evaluation:** Never assume split strategies or optimal thresholds. Claims are validated with 1,000-resample non-parametric bootstrap confidence intervals.
3. **Evidence-Driven Business Decision:** Machine learning metrics (PR-AUC, ROC-AUC) are intermediate; the system solves for optimal operational thresholds via a 12-step cost matrix workflow across held-out test data.
4. **Permanent $0 Infrastructure:** Every hosted component runs on free tiers (Supabase, Render, Vercel) and local tooling (Ollama for local LLMs). Zero paid API keys or expiring trial dependencies.
5. **Separation of Concerns:**
   - **Model** $\rightarrow$ Computes predicted probability and assigns risk tier (`LOW` / `MEDIUM` / `HIGH`).
   - **TreeSHAP** $\rightarrow$ Extracts empirical risk drivers and mitigating factors (authoritative evidence).
   - **GenAI Layer** $\rightarrow$ Formats SHAP evidence into human analyst briefings, verified by the automated Grounding Validator. The system never claims *"AI decided this transaction is fraud"*.

---

## 3. Data Integrity & Investigation Discoveries

| Investigation Area | Confirmed Truth | Engineering Action Taken |
|---|---|---|
| **TransactionDT Type** | Relative delta in seconds from undisclosed origin $[86,400 – 15,811,131]$ (182 days / 26 weeks). | Treated strictly as relative intervals. Never anchored to an invented calendar timestamp. Diurnal ($24\text{h}$) and day-of-week ($7\text{d}$) cyclical features engineered. |
| **Validation Split** | Random split ($74.7\%$ entity overlap) leaks identity and inflates PR-AUC by $+24.8\%$. | Enforced strict **temporal split at $TransactionDT \le 12,192,854$** ($80\text{th percentile}$, $67.6\%$ realistic entity overlap). Train: $472,432$ rows; Test: $118,108$ rows. |
| **Feature Audits** | Raw $C1$ is identical to velocity ($|r|=1.000$); raw $D1$ is identical to time-since-last-transaction ($|r|=1.000$). | Utilized $C1$ and $D1$ directly. Built $24$ non-redundant composite features ($|r| < 0.25$ to $C1/D1$). |
| **V-Feature Collinearity** | $162$ pairs of $V$-features have $|r| \ge 0.98$ (e.g. $V95/V101/V279$). | Consolidated into unified domain clusters in the reason code engine (`src/explainability/reason_codes.py`). |

---

## 4. Machine Learning Benchmarks (1,000-Resample 95% Bootstrap CIs)

Evaluated on held-out test partition ($N = 118,108$ transactions, $4,064$ frauds, $TransactionDT > 12,192,854$):

| Metric | Champion LightGBM | Baseline Logistic Regression | Relative Lift / Gain | 95% Bootstrap Confidence Interval |
|---|---|---|---|---|
| **PR-AUC (Primary)** | **0.5441** | 0.2746 | **+98.1% Lift** | LightGBM: `[0.5282, 0.5607]` vs. Baseline: `[0.2605, 0.2891]` |
| **ROC-AUC** | **0.9035** | 0.8092 | **+11.7% Lift** | LightGBM: `[0.8982, 0.9087]` vs. Baseline: `[0.8021, 0.8164]` |
| **Recall @ 1% FPR** | **46.63%** | 15.08% | **3.09x Capture** | LightGBM: `[44.90%, 48.24%]` (captures 1,895 frauds at 1% FPR) |
| **Recall @ 5% FPR** | **65.95%** | 41.76% | **+24.19 pp** | LightGBM: `[64.46%, 67.45%]` (captures 2,680 frauds at 5% FPR) |
| **Brier Score** | **0.0246** | 0.1782 | **-86.2% Error** | Substantially superior calibration |

### Generalization Stress Test (Unseen Entities, 0% Overlap):
On a held-out slice of completely novel card entities ($N = 10,952$, $0\%$ overlap):
- **PR-AUC:** `0.4487` ($-17.53\%$ relative decay vs. $0.5441$, yet $+63.4\%$ higher than baseline).
- **ROC-AUC:** `0.8774` ($-2.89\%$ relative decay vs. $0.9035$).
- **Recall @ 1% FPR:** `36.36%` ($-22.02\%$ relative capture reduction vs. $46.63\%$).

---

## 5. Cost-Sensitive Business Decision Summary (§4b)

All metrics derived from the 12-step cost matrix workflow on held-out test data ($N = 118,108$, Base Case: $L_{fraud} = \$200, C_{review} = \$8, C_{stepup} = \$0.50, \eta_{stepup} = 0.80$):

| Operating Policy | Cutoffs ($\tau_{med}, \tau_{high}$) | Review Rate % (Vol) | High-Tier Recall | Precision | FPR | Expected Cost | Net Savings vs. Accept All | Status |
|---|---|---|---|---|---|---|---|---|
| **No Model: Accept All** | N/A | 0.0% (0) | 0.0% | N/A | 0.0% | $812,800.00 | $0.00 | Rejected |
| **Naive Amount Rule (> $500)** | Amt > $500 | 4.08% (4,816) | 5.76% | 4.86% | 4.02% | $804,528.00 | $8,272.00 | Rejected |
| **Logistic Regression Baseline** | $p \ge 0.50$ | 22.09% (26,089) | 66.95% | 10.43% | 20.49% | $477,312.00 | $335,488.00 | Queue Overflow |
| **Candidate Policy A (Conservative)** | $\tau_{med}=0.01, \tau_{high}=0.96$ | **0.88%** (1,044) | 23.08% | **89.85%** | **0.093%** | $189,169.50 | $623,630.50 | 1% Cap Viable |
| **Candidate Policy B (Balanced)** | $\tau_{med}=0.01, \tau_{high}=0.70$ | **3.64%** (4,297) | **53.94%** | **51.01%** | **1.85%** | **$163,367.00** | **$649,433.00** | **RECOMMENDED** |
| **Candidate Policy C (Aggressive)** | $\tau_{med}=0.01, \tau_{high}=0.70$ | **3.64%** (4,297) | **53.94%** | **51.01%** | **1.85%** | **$163,367.00** | **$649,433.00** | Converges to B |

> **Authoritative Stakeholder Deliverable:** For the comprehensive 10-section narrative analysis, see the [Business Case Study](./case-study/fraud-risk-case-study.md).

---

## 6. TreeSHAP Explainability & Grounding Validator Architecture

```
                       MODEL INFERENCE PIPELINE
  Transaction Payload ──> Champion LightGBM ──> Predicted Probability (p)
                                  │
                                  ▼
                         TreeSHAP Explainer
                                  │
                                  ▼
                         Reason Code Engine
                         (162 V-Pairs Consolidated)
                                  │
                                  ▼
                        Top-5 Risk / Mitigating Codes
                                  │
                                  ▼
                      Multi-Provider LLM Prompt
                   (Local Ollama / Grok / Template)
                                  │
                                  ▼
                    Automated Grounding Validator
                   ├── Direct Number Audit
                   ├── Derived Multiples Check
                   ├── Directional Consistency
                   └── Speculation Filter
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
      [Passes Audit]                            [Fails Audit]
   100% Grounded Narrative               Fallback to Deterministic
   Delivered to UI / DB                  Verified Template
```

---

## 7. Serving Layer & REST API Contracts

The serving engine (`api/`) is built on **FastAPI** with Pydantic request validation and connection pooling to PostgreSQL on Supabase:

### Endpoints:
- `GET /health` $\rightarrow$ System health status, loaded model metadata, database connection status.
- `POST /predict` $\rightarrow$ Scores single transaction payload, calculates latency, returns risk tier, action, and TreeSHAP reason codes.
- `POST /predict/batch` $\rightarrow$ Vectorized batch scoring for up to $1,000$ transactions.
- `GET /replay?limit=1500` $\rightarrow$ Paginated replay feed over held-out test transactions and grounded narratives.
- `GET /monitoring/operational` $\rightarrow$ Live operational metrics (unlabeled volume, review rate, score deciles).
- `GET /monitoring/evaluation` $\rightarrow$ Labeled evaluation benchmarks on test replay data.

### Request / Response Schema Example (`POST /predict`):
```json
// Request
{
  "TransactionID": 3577540,
  "TransactionAmt": 290.0,
  "ProductCD": "W",
  "card1": 13926,
  "card2": 360,
  "addr1": 315,
  "P_emaildomain": "gmail.com"
}

// Response
{
  "transaction_id": 3577540,
  "predicted_probability": 0.042,
  "predicted_risk_tier": "MEDIUM",
  "decision_action": "STEP_UP_AUTH",
  "recommended_workflow": "Trigger step-up 3D-Secure / OTP verification challenge",
  "latency_ms": 313.57,
  "explanation": {
    "top_risk_factors": [
      {
        "feature": "TransactionAmt",
        "display_name": "Transaction Amount",
        "value": "$290.00",
        "shap_value": 0.42,
        "direction": "INCREASES_RISK"
      }
    ],
    "top_mitigating_factors": [
      {
        "feature": "freq_card1",
        "display_name": "Card Historical Frequency",
        "value": "142 transactions",
        "shap_value": -0.58,
        "direction": "REDUCES_RISK"
      }
    ]
  }
}
```

---

## 8. Local Setup & Reproduction Guide

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm 9+
- Git

### 1. Clone & Setup Python Environment
```bash
git clone https://github.com/Soham-Satpute/fraud-risk-analytics.git
cd fraud-risk-analytics
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the 72-Test Automated Pytest Suite
```bash
python -m pytest -v
```
*(All 72 tests across data quality, feature engineering, models, explainability, grounding validator, business decision, and API pass in ~18 seconds)*

### 3. Launch the FastAPI Serving Engine
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at: `http://localhost:8000/docs`

### 4. Launch the Next.js Frontend Demo
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to access the **Interactive Fraud Scoring Demo** and `/methodology`.

---

## 9. Deployment Blueprint & Permanent $0 Stack

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   PERMANENT $0 HOSTED ARCHITECTURE                      │
 ├───────────────────┬───────────────────┬─────────────────────────────────┤
 │ Service           │ Free Tier Usage   │ Inactivity & Cold-Start Behavior│
 ├───────────────────┼───────────────────┼─────────────────────────────────┤
 │ PostgreSQL        │ Supabase Free     │ Auto-pauses after 7 days idle.  │
 │ (Audit Store)     │ (<500MB DB)       │ API buffer maintains 100% uptime│
 ├───────────────────┼───────────────────┼─────────────────────────────────┤
 │ FastAPI Backend   │ Render Free Tier  │ Sleeps after 15m of inactivity. │
 │ (Inference API)   │ (512MB RAM)       │ Frontend has static fallback.   │
 ├───────────────────┼───────────────────┼─────────────────────────────────┤
 │ Next.js Demo UI   │ Vercel Free Tier  │ Perpetual serverless uptime.    │
 │ (Frontend)        │ (Edge Network)    │ Fast global CDN distribution.   │
 └───────────────────┴───────────────────┴─────────────────────────────────┘
```

---

## 10. Repository Structure

```
fraud-risk-analytics/
├── .github/workflows/ci.yml                  # Continuous Integration automated test workflow
├── README.md                                 # Technical architecture, ML internals & developer guide
├── case-study/
│   └── fraud-risk-case-study.md              # Authoritative 10-section stakeholder case study (§12)
├── docs/
│   ├── 01_data_integrity_investigation.md    # Week 1: Overlap, temporal split, V/D/C feature audit
│   ├── 02_data_quality_notes.md              # Week 2: Automated data checks & Postgres segregation
│   ├── 03_business_decision_and_threshold_analysis.md # Week 8: 12-step cost workflow & sensitivity
│   └── 04_render_deployment_guide.md         # Week 7: Render deployment & verification guide
├── notebooks/
│   ├── 01_data_integrity_investigation.ipynb
│   ├── 02_eda_and_storytelling.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_shap_and_narrative_generation.ipynb
├── src/
│   ├── data/                                 # Data loading & Supabase seeding utilities
│   ├── features/                             # Leakage-free feature pipeline transformer
│   ├── eda/                                  # Statistical insights engine & Wilson Score CIs
│   ├── models/                               # Training, evaluation & 12-step threshold solver
│   ├── explainability/                       # TreeSHAP explainer & reason code consolidator
│   └── validation/                           # Data quality engine & Grounding Validator
├── api/
│   ├── main.py                               # FastAPI serving application
│   ├── routes.py                             # /predict, /health, /replay, /monitoring endpoints
│   ├── schemas.py                            # Pydantic request/response validation schemas
│   └── db.py                                 # Supabase client with non-blocking fallback buffer
├── frontend/                                 # Next.js 14 (React, Vanilla CSS dark fintech UI)
│   ├── src/app/page.tsx                      # Page 1: Interactive Fraud Scoring Demo
│   ├── src/app/methodology/page.tsx          # Page 2: Methodology & Analytics Summary
│   └── src/components/                       # ScoreGauge, ReasonCodesPanel, NarrativeCard, Sandbox
├── sql/
│   ├── schema.sql                            # predictions, model_runs, demo_replay DDL
│   └── analytics_queries.sql                 # Daily trends, PR/FPR, drift monitoring queries
├── dashboard/
│   ├── powerbi_setup_guide.md                # DirectQuery / Import setup guide
│   └── dax_measures_and_model_health.md      # DAX formulas & Model Health tile specifications
└── tests/
    ├── test_api.py                           # 14 Pytest API unit & integration tests
    ├── test_business_decision.py             # 6 Pytest cost matrix & sensitivity tests
    ├── test_data_quality.py                  # 20 Pytest data validation tests
    ├── test_eda_insights.py                  # 7 Pytest Wilson CI & story tests
    ├── test_explainability.py                # 6 Pytest reason code & policy tests
    ├── test_feature_engineering.py           # 8 Pytest leakage & math tests
    ├── test_grounding_validator.py           # 6 Pytest GenAI grounding audit tests
    └── test_models.py                        # 5 Pytest evaluation metric tests
```

---

## 11. Defensible Limitations & Next Steps

1. **Entity Proxy Approximations:** Identity columns (`card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`) serve as approximate entity proxies, not ground-truth verified consumer IDs. Cold-start entities experience $\sim 17.5\%$ PR-AUC decay ($0.4487$).
2. **Missing Temporal Origin:** `TransactionDT` represents relative seconds from an undisclosed origin. In enterprise production, absolute timestamps with seasonal calendar and timezone features would be integrated.
3. **Enterprise Production Next Steps:** Real-time sliding-window feature stores (Redis/Feast), sub-50ms streaming event pipelines (Kafka/Flink), and automated Kolmogorov-Smirnov drift alerts triggering scheduled model retraining.
