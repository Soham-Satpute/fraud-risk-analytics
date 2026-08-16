# AGENTS.md — Repository Guidance & Agent Operating Instructions

> **Project Name:** Fraud Risk Analytics & Detection System  
> **Supporting Capability:** Grounded GenAI Analyst Explanations  
> **Source Plan:** `fintech-fraud-analytics-plan-v6.md`  
> **Objective:** Deliver a rigorous, defensible, end-to-end fintech fraud risk analytics and modeling system with statistical rigor, honest evaluation, grounded explainability, lightweight deployment, and stakeholder business recommendations — built entirely on a $0 free-tier stack.

---

## 1. Project Philosophy & Core Identity

1. **Investigation & Modeling are the Product**: The core differentiator is investigative rigor (entity overlap, temporal leakage, V/D/C feature audits, class imbalance handling, cost-sensitive threshold optimization). GenAI is only a supporting explanation layer, not the headline.
2. **Honest & Defensible Evaluation**: Never assume split strategies or optimal thresholds. Frame investigation questions neutrally (e.g., *"How do entity overlap and temporal structure affect validation reliability?"*). Validate claims with bootstrapped confidence intervals.
3. **Evidence-Driven Business Decision**: Machine learning metrics (PR-AUC, ROC-AUC) are intermediate; the end deliverable must translate model scores into an operating threshold, financial cost impact, review volume, and actionable stakeholder recommendation (§4a/§4b). Concluding "not worth deploying over baseline" is a valid outcome.
4. **Permanent $0 Infrastructure**: Every hosted component runs on free tiers (Supabase, Render, Vercel, Kaggle) and local tooling (Ollama for local LLMs). Never introduce paid API keys or expiring trial dependencies.

---

## 2. Technology Stack & Architectural Boundaries

| Layer | Approved Technology | Boundaries & Strict Rules |
|---|---|---|
| **Dataset** | IEEE-CIS Fraud Detection (Kaggle) | **Only dataset allowed.** No PaySim or synthetic demo data. |
| **Data Processing & ML** | `pandas`, `numpy`, `scikit-learn`, `XGBoost`, `LightGBM` | Local parquet/CSV for raw/training data. Class-weighting is default for imbalance (ablate against threshold-moving/SMOTE with justification). Baseline: Logistic Regression. |
| **Data Quality & Testing** | `pytest` | Standalone repeatable test suite in `tests/data_quality_checks.py` (schema, dtypes, nulls, range validity, duplicates). No heavy Great Expectations setup. |
| **Database** | **PostgreSQL on Supabase** (Free Tier) | **Serving/logging layer ONLY** (`predictions`, `model_runs`, small demo replay slice of a few hundred to thousand rows). **NEVER load full raw or training data into Postgres.** |
| **Explainability** | `SHAP` | TreeExplainer/LinearExplainer aggregated into top-5 reason codes per transaction. |
| **GenAI / LLM Layer** | **Local Model via Ollama** (e.g., Llama 3.2 3B, Phi-3-mini) | **Offline one-time batch generation only** (temp 0) for the held-out demo set; stored in Postgres. **Zero paid APIs.** |
| **Grounding Validator** | Custom Python script / pytest | **Mandatory companion to the LLM layer.** Verifies that every generated narrative contains only features/values present in the SHAP evidence. (If LLM narratives exist, the validator MUST exist). |
| **Backend API** | `FastAPI` (Python) | Single `/predict` endpoint (scores, returns reason codes, logs to Supabase). Deployed on Render free tier. |
| **Frontend Demo** | `Next.js` (React, deployed on Vercel) | **Strictly 2 pages max**: (1) Scoring/Demo replay page, (2) Methodology & Analytics summary. No auth, no routing complexity. |
| **BI & Analytics** | `Power BI Desktop` (Export / Publish to Web) + SQL | Connects to Postgres for fraud trends, PR/FPR tracking, cost-saved range (with explicit assumptions), and a **Model Health** tile. |

---

## 3. Strict Scope Boundaries & Anti-Patterns

### ❌ FORBIDDEN (DO NOT IMPLEMENT)
- **NO PaySim or secondary datasets**: Stick exclusively to IEEE-CIS.
- **NO RAG Chatbots / Conversational AI**: The LLM layer is strictly for offline translation of SHAP reason codes into templated analyst summaries.
- **NO Live/Per-Request Paid LLM API Calls**: Do not use OpenAI/Anthropic APIs or live LLM inference in the web app.
- **NO Heavy Infrastructure**: Kafka, Airflow, Spark, Kubernetes, Docker Swarm, Terraform, or complex distributed pipelines are out of scope.
- **NO Raw Data in Postgres**: Do not load raw IEEE-CIS tables into Supabase. Postgres is exclusively for the API serving and audit logging layer.
- **NO Default 0.5 Cutoff / Automatic Blocking**: Thresholds must be chosen via the 12-step cost matrix workflow (§4a). Never recommend unconditional auto-blocking without review tiers.
- **NO Unvalidated GenAI Output**: Never ship LLM narratives without the grounding validation test suite.

---

## 4. Locked Claim & Terminology Standards

When generating documentation, comments, API responses, or UI copy, **agents must strictly adhere to the following phrasing rules**:

| ❌ Forbidden / Tempting Phrase | ✅ Required Honest Phrasing |
|---|---|
| "AI Fraud Detection Platform" | **"Fraud Risk Analytics & Detection System"** (with Grounded GenAI Analyst Explanations as a supporting capability) |
| "Real-time fraud detection" | **"Simulated real-time inference — replays held-out transactions through the deployed model"** |
| "Production-ready system" | **"Deployment-ready portfolio demo"** |
| "AI-powered explanations" | **"Model explanations generated offline by a local LLM, grounded strictly to SHAP evidence"** |
| "Explainable AI / Causal explanation" | **"SHAP-based feature attribution, aggregated into reason codes"** |
| "Fraud prevention system" | **"Fraud risk detection/scoring system"** |
| "Client identity reconstruction" | **"Approximate, correlation-based entity identifiers (not a confirmed ground-truth key)"** |
| "Completely free / zero-cost forever" | **"Built entirely on free tiers of hosted services, each with its own usage limits and inactivity behavior"** |
| "Production MLOps monitoring" | **"Basic observability metrics (volume, score distribution, precision/recall when labels are available)"** |
| "Deployed ML system" | **"Portfolio-scale deployed demo"** |

---

## 5. Repository Structure & Deliverables Map

```
fraud-risk-analytics/
├── README.md                                 # Technical audience: Architecture, ML internals, setup, API, tradeoffs
├── case-study/
│   └── fraud-risk-case-study.md (or .pdf)    # Stakeholder audience: 10-section business case study (§12)
├── docs/
│   ├── 01_data_integrity_investigation.md    # Week 1: Entity overlap, temporal split analysis, V/D/C audit
│   ├── 02_data_quality_notes.md              # Week 2: Automated data checks & Postgres layer rationale
│   └── 03_business_decision_and_threshold_analysis.md # Week 8: 12-step workflow (§4a) & decision summary (§4b)
├── notebooks/
│   ├── 01_data_integrity_investigation.ipynb
│   ├── 02_eda_and_storytelling.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_shap_and_narrative_generation.ipynb
├── src/
│   ├── data/                                 # Data loading & preprocessing utilities
│   ├── features/                             # Velocity, amount z-score, time-since-last, frequency features
│   ├── models/                               # Logistic Regression, XGBoost/LightGBM training & cost-eval logic
│   ├── explainability/                       # SHAP reason codes & Ollama prompt templating
│   └── validation/                           # Grounding validator logic
├── api/
│   ├── main.py                               # FastAPI application
│   ├── routes.py                             # /predict, /health, /replay endpoints
│   ├── schemas.py                            # Pydantic request/response schemas
│   └── db.py                                 # Supabase Postgres connection & logging
├── frontend/                                 # Next.js (2 pages max: /demo and /methodology)
│   ├── app/ (or pages/)
│   └── components/
├── sql/
│   ├── schema.sql                            # predictions, model_runs, demo_transactions tables
│   └── analytics_queries.sql                 # Fraud trends, review volumes, model drift queries
├── dashboard/
│   └── fraud_analytics_dashboard.pbix        # Power BI report file + screenshots
├── tests/
│   ├── test_data_quality.py                  # Pytest automated checks on data batches
│   └── test_grounding_validator.py           # Pytest checks on LLM narratives vs SHAP evidence
└── models/                                   # Serialized model artifacts (.joblib / .json) + metadata
```

---

## 6. Execution Phases & Guidelines for Coding Agents

### Phase 1: Data Integrity & Feature Audit (Docs & Notebooks) — ✅ COMPLETE

**Confirmed facts — use in all downstream code, docs, and comments without re-investigation:**

| Fact | Value |
|---|---|
| `TransactionDT` type | Relative delta in seconds from undisclosed origin. Range `[86,400 – 15,811,131]`. Span: 182 days (26 weeks). Never treat as absolute timestamp. |
| Dataset shape (post-merge) | 590,540 rows × 434 columns |
| Fraud rate | 3.499% (20,663 fraud / 569,877 legitimate) |
| Identity join coverage | 23.8% — identity features are sparse; handle missingness explicitly |
| Fraud rate temporal drift | 3.40% (first half) → 3.61% (second half) |
| Chosen validation strategy | **Temporal split at TransactionDT ≤ 12,192,854** (80th percentile). Train: 472,432 rows. Test: 118,108 rows. |
| Entity proxy columns | `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain` → 94,846 unique proxies |
| Entity overlap (temporal) | **67.6%** — selected as realistic; random (74.7%) avoided due to metric inflation |
| Supplementary benchmark | Grouped entity split (0% overlap) used once at Week 5 evaluation as generalisation lower bound |
| D1 | **= time-since-last-transaction** (\|r\|=1.000 with planned feature). Use D1 directly. Do not rebuild. |
| C1 | **= transaction velocity proxy** (\|r\|=1.000 with planned feature). Use C1 directly. Do not rebuild. |
| V-feature missingness | 7 distinct clusters. 339 V-columns total (up to 85% missing in sparse clusters). |
| V near-duplicate pairs | 162 pairs with \|r\| ≥ 0.98. Top predictors: V257, V201, V246, V200 (\|r\|=0.26–0.28 with fraud). |
| Unstable D-features (PSI>0.10) | D4, D6, D10, D14, D15 — structural drift under temporal split; flag for monitoring, do not drop. |

### Phase 2: Repeatable Data Quality & Serving Schema — ✅ COMPLETE

**Confirmed deliverables & implemented components:**
- **Automated Data Quality Engine (`src/validation/data_quality.py`)**: Reusable Python validation framework & CLI enforcing schema integrity, dtypes, binary target validity (`isFraud ∈ {0, 1}` with 0% nulls), `TransactionID` uniqueness, physical/domain range bounds, categorical domain sets, and temporal span bounds.
- **Pytest Automated Test Suite (`tests/test_data_quality.py`)**: 20 passing unit & integration tests covering synthetic corrupted batches (duplicate IDs, negative amounts, invalid labels, missing columns) and verifying live parquet temporal split properties (`TransactionDT ≤ 12,192,854` for 472,432 train rows vs `> 12,192,854` for 118,108 test rows).
- **PostgreSQL Serving Schema (`sql/schema.sql`)**: Production-grade DDL defining `predictions`, `model_runs`, and `demo_replay` tables with B-tree and partial indexes (`WHERE predicted_risk_tier = 'HIGH'`) tailored for Supabase's free tier (<500MB).
- **BI & Monitoring SQL Queries (`sql/analytics_queries.sql`)**: Analytical queries for daily volume/fraud rates, confusion matrix / PR / FPR metrics, high-risk review backlog, and score decile drift monitoring.
- **Held-Out Demo Slice Extractor (`src/data/make_demo_slice.py`)**: Curated 1,500 held-out test transactions (`TransactionDT > 12,192,854`) in lightweight Parquet (517 KB) and JSON seed (418 KB) formats.
- **Engineering Documentation (`docs/02_data_quality_notes.md`)**: Rationale for storage segregation (raw data stays local columnar parquet; Postgres strictly for serving/logging) and data quality invariants.

### Phase 3: Feature Engineering & Baseline Modeling

**Feature Engineering Deliverables — ✅ COMPLETE:**
- **Leakage-Free Feature Pipeline (`src/features/engineer.py`)**: `FraudFeaturePipeline` transformer fitting frequency encodings and group statistics strictly on Train partition (`TransactionDT ≤ 12,192,854`).
- **Feature Build Automation (`src/features/build_features.py`)**: Generates `data/processed/train_features.parquet` (472,432 rows, 74.9 MB) and `data/processed/test_features.parquet` (118,108 rows, 19.6 MB) with 24 newly engineered features (458 total features).
- **Feature Unit Test Suite (`tests/test_feature_engineering.py`)**: 8 passing pytest unit tests verifying mathematical correctness, cyclical bounds $[-1, 1]$, zero temporal leakage, and pipeline serialization round-trip.
- **Cross-Correlation Audit**: Confirmed all 24 newly engineered features maintain $|r| < 0.85$ with $C1$ and $D1$ (maximum $|r| = 0.243$), guaranteeing additive non-redundant predictive signal.
- **Visual Storytelling (`notebooks/03_feature_engineering.ipynb`)**: EDA and correlation matrix notebook.

| Feature | Decision | Shipped Implementation |
|---|---|---|
| Transaction velocity | ❌ Skip — use **C1** directly | C1 is identical (\|r\|=1.000) |
| Time since last transaction | ❌ Skip — use **D1** directly | D1 is identical (\|r\|=1.000) |
| Amount z-score by card/merchant | ✅ Built | `amt_zscore_card1`, `amt_diff_mean_card1`, `amt_ratio_mean_card1`, `amt_zscore_card1_addr1`, `amt_zscore_email` |
| Log-transformed TransactionAmt | ✅ Built | `log_TransactionAmt` = $\log(1 + \text{TransactionAmt})$ |
| Merchant/category frequency | ✅ Built | `freq_card1`, `freq_card2`, `freq_addr1`, `freq_ProductCD`, `freq_R_emaildomain` |
| Hour-of-day cycle | ✅ Built | `hour_sin`, `hour_cos` (24-hour diurnal cycle) |
| Day-of-week cycle | ✅ Built | `dow_sin`, `dow_cos` (7-day weekly cycle) |
| Email consistency | ✅ Built | `email_match_flag`, `null_P_email`, `null_R_email` |

**Next Step (Baseline Modeling):**
- Train **Logistic Regression** baseline first (on merged feature set including D1, C1, key V-features).
- Train **XGBoost / LightGBM** with `scale_pos_weight` (class-weighting).
- Compute PR-AUC, ROC-AUC, and **Recall at fixed False Positive Rates (e.g., FPR=1%, 5%)** with 95% bootstrapped confidence intervals (1,000 resamples).

### Phase 4: Explainability & Grounded Narrative Generation
- Compute TreeSHAP values for held-out predictions. Extract top-5 positive and negative contributing features as reason codes.
- **V-feature collinearity handling:** 162 near-duplicate V-feature pairs (|r|≥0.98) exist. When aggregating SHAP reason codes, consolidate near-duplicate pairs (e.g., V95/V101/V279/V293 cluster) into a single reason code — do not surface four near-identical features as separate reasons to a stakeholder.
- Use a local Ollama model (e.g., `llama3.2:3b`) with `temperature=0` to convert SHAP reason codes into concise, structured analyst summaries.
- Run `tests/test_grounding_validator.py`: ensure regex/entity extraction confirms that every numeric value and feature mentioned in the narrative matches the SHAP payload.

### Phase 5: Business Decision Workflow (§4a & §4b)
- Execute the full 12-step workflow below end-to-end on the deployed model's held-out predictions (already logged in Postgres from Phase 6 — no new data pipeline needed).
- **The recommendation is not decided in advance.** The workflow must be run for real, on real numbers, and must be allowed to conclude the model doesn't justify deployment — that is a valid, reportable outcome.
- Populate `docs/03_business_decision_and_threshold_analysis.md` with concrete numbers and the completed §4b template (with real numbers, never placeholders).

### Phase 6: FastAPI Backend & Next.js Frontend
- **FastAPI**: Clean REST endpoints with Pydantic validation, structured error handling, and async database logging to Supabase.
- **Next.js**: Clean, responsive, high-aesthetic UI strictly constrained to 2 views:
  1. *Simulated Stream & Scoring Page*: Replays held-out transactions, calls `/predict`, displays probability gauge, SHAP reason codes, and grounded narrative.
  2. *Methodology & Performance Page*: Summary of validation split findings, cost curves, and confidence intervals.

### Phase 7: Deliverables Separation (README vs Case Study)
- **Technical README (`README.md`)**: Targeted at engineers & technical interviewers. Details data pipelines, ML architecture, API contracts, deployment instructions, and limitations.
- **Business Case Study (`case-study/fraud-risk-case-study.md`)**: Targeted at hiring managers, analysts, and business stakeholders. Follows the locked 10-section structure from §12. Populated ONLY with real analysis numbers. **Do not duplicate content between the two — they serve different audiences.**

---

## 7. Quality Standards & Coding Conventions

1. **Deterministic & Reproducible**: Set explicit random seeds (`random_state=42`) across all split, training, and bootstrap routines. Pin dependencies in `requirements.txt` / `package.json`.
2. **Clean Python Architecture**: Type hints on all functions (`src/`, `api/`), comprehensive docstrings, modular organization (no 1000-line monolithic scripts).
3. **Robust Error Handling**: Graceful fallback in FastAPI and Next.js when database or network latency occurs (especially Render/Supabase cold-starts).
4. **No Premature Claims**: Do not write placeholder numbers as final results. Clearly mark placeholder metrics until modeling scripts produce final verified values.

---

## 8. Business Decision Workflow — Full 12-Step Spec (§4a)

This step turns a trained model into an actual recommendation. Runs on artifacts that already exist by Phase 6 (held-out predictions in Postgres) — no new infrastructure.

1. Establish realistic business cost assumptions (average loss per missed fraud, average cost of manual review — sourced or clearly labeled as assumptions where no public figure exists).
2. Establish operational constraints — realistic manual-review capacity (e.g., as a % of daily transaction volume a small fraud team could plausibly review).
3. Evaluate the model across a range of thresholds (not just one default cutoff).
4. Calculate fraud capture (recall) at each threshold.
5. Calculate false-positive rate at each threshold.
6. Calculate expected manual-review volume at each threshold.
7. Estimate expected financial cost at each threshold, combining missed-fraud cost and review cost.
8. Compare the best candidate threshold's cost against the baseline (naive rule or default-threshold Logistic Regression) — quantify the improvement, or the lack of one.
9. Run sensitivity/scenario analysis: fraud cost ±20%, false-positive cost changes, review capacity at 5% vs. 10%, and at least one alternate threshold choice.
10. Determine whether the recommendation is robust to reasonable changes in assumptions — if the "right" threshold changes with capacity or cost assumptions, that dependency is itself a finding, not a flaw to hide.
11. Select the recommended operating point (or state that no threshold clears the bar over baseline, if that's what the evidence shows).
12. Translate the result into a plain-language stakeholder recommendation (§4b), including what would change it.

> **Do not assume a single universally optimal threshold exists.** If the appropriate choice depends on review capacity or cost assumptions, report that as a finding rather than picking one number and hiding the dependency.

---

## 9. Final Decision & Business Recommendation — Required Output Spec (§4b)

The project **must produce this as an actual artifact** (feeds directly into `docs/03_business_decision_and_threshold_analysis.md` and the Case Study §07–09). Never end with a bare model-performance number.

**Decision summary must contain all 12 fields:**
1. Key analytical findings (from the investigation and modeling, in plain language)
2. Recommended threshold
3. Fraud capture at that threshold
4. False-positive rate at that threshold
5. Expected manual-review volume
6. Estimated financial impact, presented as a **range with stated assumptions** (never a single unqualified figure)
7. Comparison against the baseline (quantified — how much better, if at all)
8. Operational recommendation — how the model should actually be used
9. Major assumptions the recommendation depends on
10. Limitations
11. Monitoring requirements going forward
12. Conditions that would trigger re-evaluation of the threshold or the model itself

> **Critical rule:** The recommendation must **not** default to automatic transaction blocking. A tiered structure (high-risk → manual review, medium-risk → step-up checks, low-risk → normal processing) is one example, not a template — the analysis decides the structure, not the other way around.

---

## 10. Final Locked Scope (§11)

**MUST HAVE**
- Data integrity investigation (neutral framing, documented finding)
- Leakage/entity-overlap analysis, `V`/`D`/`C` feature audit
- Feature engineering (pandas)
- Logistic Regression baseline + XGBoost (class-weighted)
- Cost-sensitive threshold selection
- PR-AUC / recall@fixed-FPR evaluation
- SHAP reason codes
- FastAPI backend
- PostgreSQL (serving/logging layer only)
- Deployed frontend (Next.js on Vercel, two pages max)
- Business Decision Workflow (§4a): threshold sweep, quantified baseline comparison, sensitivity/scenario analysis
- Final Decision & Business Recommendation artifact (§4b), with the possibility of concluding the model doesn't justify deployment
- Business Case Study (§12), separate from the README, populated with real results only
- Strong README / written case study

**SHOULD HAVE**
- Automated data-quality tests
- Power BI dashboard
- Basic post-deployment monitoring
- SQL analytics queries against the predictions table
- Reproducibility (pinned environment, fixed seeds)
- Statistical confidence framing wherever a business claim is made: (a) EDA insights, (b) headline recall@FPR metric with bootstrapped or Wilson CI, (c) Power BI cost-saved estimate stated as a range with explicit assumptions — bare point estimates are not enough

**OPTIONAL** *(if cut, cut the whole unit together)*
- Local LLM (Ollama) grounded narrative layer — **if built, the grounding validator ships as part of the same unit; it is NOT separately optional**
- Grounding validator — conditionally required; only skippable by skipping the LLM layer entirely
- GitHub Actions CI
- Calibration analysis (reliability diagram)

**REMOVE**
- RAG chatbot
- PaySim
- Kafka, Airflow, Kubernetes
- Unused Supabase features (auth, storage, realtime)
- Any live or paid LLM API call — including for one-time offline generation

---

## 11. Business Case Study — 10-Section Locked Structure (§12)

The case study is a **distinct artifact from the README** — answers "what problem → what did we investigate → what did we discover → what decision should be made → why does it matter?" for a non-ML audience. It is **not** another technical walkthrough.

> **Rule:** Build the structure now. Populate it **only** after Week 8's real analysis is complete. Never ship placeholder numbers — every `[X]`/`[Y]` must be replaced with actual findings. Target length: 8–12 pages.

| Section | Content |
|---|---|
| **01 — Executive Summary** | One page: business problem, approach, major finding, recommended action, business impact. Understandable without ML background. |
| **02 — Business Problem** | Why fraud detection matters, who the stakeholder is, what decision they need to make, what operational constraints exist. |
| **03 — Data Investigation** | Dataset characteristics, missingness, class imbalance, temporal structure, entity overlap, leakage investigation. Centered on: *what did we discover that changed our methodology?* |
| **04 — Modeling Approach** | Baseline, main model, feature engineering, class weighting, validation strategy, and why. Not a tutorial — a decisions log. |
| **05 — Model Results** | PR-AUC, precision/recall, recall@fixed-FPR (with CI), confusion matrix, calibration if used, and direct baseline comparison. |
| **06 — Explainability** | One real transaction example: fraud probability, top SHAP reasons, human-readable explanation, and grounding safeguard description (if LLM layer used). |
| **07 — Business Impact & Threshold Decision** | Threshold comparison, fraud captured, false positives, review volume, expected cost, financial impact, stated Recommendation sourced from §4b. |
| **08 — Sensitivity Analysis** | What happens as fraud cost, false-positive cost, and review capacity change; answers whether the recommendation is robust. |
| **09 — Final Business Recommendation** | What the business should do, how the model should be used, what different risk levels trigger, and why (not automatic blocking by default). |
| **10 — Limitations & Next Steps** | IEEE-CIS limitations, cost-assumption limits, label limits, approximate entity reconstruction, differences from real production data, free-tier infra limits, what a real deployment needs next. |

---

## 12. README vs. Case Study — Audience & Content Split (§13)

Two deliverables, two audiences — **do not duplicate one inside the other.**

| | **README** | **Business Case Study** |
|---|---|---|
| **Audience** | Developers, technical interviewers, engineers | Recruiters, hiring managers, data analysts, business stakeholders |
| **Focus** | Architecture, code, setup, methodology, ML internals, API, database, deployment, testing, monitoring | Problem, investigation, findings, model value, business impact, decision, recommendation, limitations |
| **Depth on ML internals** | Full | Summarized — enough to trust the conclusion, not enough to reproduce the model |
| **Depth on business decision** | Linked, summarized | Full — this is the case study's centerpiece (§12 §07–09) |

