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
│   ├── features/                             # Frequency, amount z-scores, cyclical, email features
│   ├── eda/                                  # Statistical insights & hypothesis testing
│   ├── models/                               # Logistic Regression, XGBoost/LightGBM training & cost-eval logic
│   ├── explainability/                       # SHAP reason codes & Ollama prompt templating
│   └── validation/                           # Data quality engine & Grounding validator logic
├── api/
│   ├── main.py                               # FastAPI application
│   ├── routes.py                             # /predict, /health, /replay endpoints
│   ├── schemas.py                            # Pydantic request/response schemas
│   └── db.py                                 # Supabase Postgres connection & logging
├── frontend/                                 # Next.js (2 pages max: /demo and /methodology)
│   ├── app/ (or pages/)
│   └── components/
├── sql/
│   ├── schema.sql                            # predictions, model_runs, demo_replay tables
│   └── analytics_queries.sql                 # Fraud trends, review volumes, model drift queries
├── dashboard/
│   └── fraud_analytics_dashboard.pbix        # Power BI report file + screenshots
├── tests/
│   ├── test_data_quality.py                  # Pytest automated checks on data batches
│   ├── test_feature_engineering.py           # Pytest unit tests for leakage-free feature pipeline
│   ├── test_eda_insights.py                  # Pytest tests for Wilson CIs and statistical stories
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

**EDA & Storytelling Deliverables (Week 4) — ✅ COMPLETE:**
- **Statistical Insights Engine (`src/eda/insights.py`)**: Computes 5 concrete fraud stories with 95% Wilson Score Confidence Intervals, Risk Ratios ($RR$), and financial loss metrics strictly on the training partition ($N=472,432$).
- **Storytelling Notebook (`notebooks/02_eda_and_storytelling.ipynb`)**: Publication-quality visual analysis and executive callouts covering diurnal attack windows (1.36x risk), self-transfer recipient anomalies (3.30x risk, 9.29% fraud), relative card z-scores (1.32x risk), product channel risk disparities, and the identity capture paradox (3.55x risk).
- **Statistical Manifest (`data/processed/eda_insights_summary.json`)**: Machine-readable statistics for downstream case study quotations.
- **Unit Test Suite (`tests/test_eda_insights.py`)**: 7 passing tests validating Wilson interval bounds, edge cases, and story consistency (35/35 total repository tests passing).

**Baseline & Tree Modeling Deliverables (Week 5) — ✅ COMPLETE:**
- **Evaluation Toolkit (`src/models/evaluation.py`)**: Rank metrics (PR-AUC, ROC-AUC, Log-Loss, Brier score), operational threshold solver (`calculate_recall_at_fixed_fpr`), 1,000-resample non-parametric bootstrap confidence interval engine (`bootstrap_metric_confidence_intervals`), and fine-grained threshold sweeper (`generate_threshold_sweep`).
- **End-to-End Training Pipeline (`src/models/train.py`)**: Trains Logistic Regression baseline (StandardScaler + SimpleImputer), Champion LightGBM (`scale_pos_weight=27.46`, 200 trees), and Ablation unweighted LightGBM strictly on $N=472,432$ train set.
- **Held-Out Test Set Benchmark ($N=118,108$ post-cutoff):**
  - **PR-AUC (Primary):** Champion `0.5441` (95% CI: `[0.5282, 0.5607]`) vs Baseline `0.2746` (+98.1% lift)
  - **ROC-AUC:** Champion `0.9035` (95% CI: `[0.8982, 0.9087]`) vs Baseline `0.8092` (+11.7% lift)
  - **Recall @ 1% FPR:** Champion `46.63%` (95% CI: `[44.90%, 48.24%]`) vs Baseline `15.08%` (3.09x capture)
  - **Recall @ 5% FPR:** Champion `65.95%` (95% CI: `[64.46%, 67.45%]`) vs Baseline `41.76%` (+24.19 pp)
- **Generalization Stress Test (Unseen Entities, $N=10,952$, 0% Overlap):** PR-AUC `0.4487`, ROC-AUC `0.8774`, Recall @ 1% FPR `36.36%` (~17.5% degradation on brand-new unseen entities).
- **Serialized Model Artifacts:** `models/champion_model.joblib`, `models/baseline_logistic_regression.joblib`, and `models/model_metrics.json`.
- **Unit Test Suite (`tests/test_models.py`)**: 5 passing pytest unit tests covering metric accuracy, fixed FPR thresholding, bootstrap CI monotonicity, and threshold sweeps (40/40 total repository tests passing).
- **Modeling Notebook (`notebooks/04_model_training_and_evaluation.ipynb`)**: Head-to-head comparison, top-20 gain feature importances, unseen entity stress test, and review queue sizing sweep.

### Phase 4: Explainability & Grounded Narrative Generation (Week 6) — ✅ COMPLETE

**Confirmed deliverables & implemented components:**
- **Reason Code & Collinearity Consolidation Engine (`src/explainability/reason_codes.py`)**: Consolidates 162 near-duplicate collinear $V$-feature pairs ($|r| \ge 0.98$ from Week 1) into unified driver clusters (e.g. *Payment Activity Volume Cluster* for $V95/V101/V279/V293$). Maps features to human-readable domain descriptors with units and delta comparisons. Enforces the predefined business action policy (`APPROVE`, `STEP_UP_AUTH`, `MANUAL_REVIEW`).
- **TreeSHAP Attribution Engine (`src/explainability/shap_explainer.py`)**: High-performance TreeSHAP wrapper for Champion LightGBM supporting interactive single-transaction inference-time explanations and vectorized batch explanations across DataFrames.
- **Multi-Provider LLM Narrative Layer (`src/explainability/llm_client.py` & `src/explainability/narrative_generator.py`)**: Tiered provider hierarchy: Ollama (Primary local model) $\rightarrow$ Deterministic Template (Baseline guarantee) $\rightarrow$ Grok/xAI API (Optional experiment). Generates structured 4-section analyst assessments with automatic fallback-on-rejection.
- **Automated Grounding Validator Engine (`src/validation/grounding_validator.py`)**: Audits narratives against direct numbers, derived multiples (recalculated from baseline evidence), feature existence, directional consistency, and anti-speculation rules.
- **Offline Batch Generation Pipeline (`src/explainability/batch_generate_narratives.py`)**: Enriched the 1,500 held-out test transactions in `data/processed/demo_replay_slice.parquet` and `data/processed/demo_replay_slice.json`. Generated empirical audit manifest `data/processed/grounding_validation_report.json` (93.47% empirical initial pass rate, 100.0% final verified rate after fallback substitution).
- **Unit Test Suites (`tests/test_explainability.py` & `tests/test_grounding_validator.py`)**: 12 new passing pytest tests (52/52 total repository tests passing).
- **Visual Storytelling Notebook (`notebooks/05_shap_and_narrative_generation.ipynb`)**: Beeswarm/summary plots, collinearity consolidation demonstrations, case studies, and grounding audit visualizations.

### Phase 5: Business Decision Workflow (§4a & §4b)
- Execute the full 12-step workflow below end-to-end on the deployed model's held-out predictions (already logged in Postgres from Phase 6 — no new data pipeline needed).
- **The recommendation is not decided in advance.** The workflow must be run for real, on real numbers, and must be allowed to conclude the model doesn't justify deployment — that is a valid, reportable outcome.
- Populate `docs/03_business_decision_and_threshold_analysis.md` with concrete numbers and the completed §4b template (with real numbers, never placeholders).

### Phase 6: FastAPI Backend, Deployment & Monitoring (Week 7) — ✅ BACKEND COMPLETE
- **FastAPI Serving Engine (`api/main.py`, `api/routes.py`, `api/schemas.py`, `api/db.py`, `api/config.py`)**:
  - Live `/predict` & `/predict/batch` endpoints: loads Champion LightGBM booster, fitted feature pipeline, and TreeSHAP explainer with reason code aggregation. Measures and returns empirical request latency (`latency_ms`, empirical $p50 \approx 313\text{ms}$).
  - Strict Pydantic validation & security guardrails: physical bound checks (TransactionAmt > 0), request body size limiter (<1MB), CORS restriction, and error message sanitization.
  - Resilient Supabase PostgreSQL client with connection pooling and non-blocking in-memory fallback buffer (guaranteeing 100% API availability during cold starts or offline development).
  - `/replay` & `/replay/{transaction_id}`: paginated streaming feed over the 1,500 held-out test transactions and grounded narratives for the portfolio demo frontend.
  - Decoupled observability endpoints: `/monitoring/operational` (unlabeled volume, score deciles, review queue) vs. `/monitoring/evaluation` (labeled precision, recall, FPR benchmark on held-out test replay).
- **Database Seeding & Migration Automation (`src/data/seed_supabase.py`)**: CLI utility to apply schema DDL (`sql/schema.sql`), seed `model_runs` benchmarks, load `demo_replay` records, and batch-replay predictions into PostgreSQL.
- **Power BI Dashboard Integration & DAX Specifications (`dashboard/`)**:
  - `dashboard/powerbi_setup_guide.md`: DirectQuery / Import mode guide for Supabase PostgreSQL.
  - `dashboard/dax_measures_and_model_health.md`: DAX code library for operational volume, test benchmark metrics, parameterized Cost-Saved Range Model ($Cost_{saved} = [Caught_{fraud} \times L_{fraud} - Volume_{review} \times C_{review}]$ with stated assumptions), and Model Health Tile specifications.
  - `dashboard/export_analytics_extracts.py`: Automated extractor generating static CSV datasets in `dashboard/data/` for offline Power BI report authoring.
- **Deployment Assets & Verification Guide (`docs/04_render_deployment_guide.md`)**:
  - Production containerization & blueprints: `render.yaml`, `Dockerfile`, `.env.example`.
  - Comprehensive deployment verification checklist with health ping, curl tests, and cold-start handling.
- **Automated Pytest API Test Suite (`tests/test_api.py`)**: 14 passing unit and integration tests covering health, inference, batch scoring, empirical latency measurement, validation/payload size guards, replay pagination, and database degradation (66/66 total repository tests passing).
- **Next.js Frontend (Week 9)**: Clean, responsive, high-aesthetic UI strictly constrained to 2 views:
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

