# Fraud Risk Analytics & Detection System — Implementation Plan (v6, final scope)

**Project name:** *Fraud Risk Analytics & Detection System* — with **Grounded GenAI Analyst Explanations** as a supporting capability inside it, not the headline. The ML model + investigation are the product; the LLM layer explains it. Avoid naming this an "AI Fraud Detection Platform" anywhere in the README/resume — that framing overstates the LLM's role and understates the actual differentiator (the investigation and modeling rigor).

**Goal:** Prove you can operate as a full-stack data analyst/scientist — rigorous investigation, modeling, explainability, honest deployment, and business communication — via one deep, defensible fintech project, built entirely on free-tier infrastructure.

**What changed from v3:** This version locks in the final scope after a structured critic review. Key decisions: **Next.js + Vercel is kept** (deliberately small — two pages max), the LLM layer moves to a **local model via Ollama** (offline generation, no paid API at all, not even for pre-generation), a **grounding validator** is added as a named optional component to make the GenAI layer more than cosmetic, and every "impressive-sounding" claim in the README gets a precise, honest rewording (see §10). §11 has the final locked scope (MUST / SHOULD / OPTIONAL / REMOVE).

**What changed from v4 (final tightening pass):** (1) Database choice locked to **Supabase only** — Neon dropped as an alternative to avoid an open decision at build time. (2) The **grounding validator is no longer independently optional** — it's now a *conditional requirement*: if the local-LLM narrative layer is built at all, the validator must ship with it (§11). (3) Statistical validation is no longer scoped to EDA claims alone — it now also covers the headline model-performance metric and the cost-saved estimate (§7, Week 7, §11). No new technologies added; this pass only removes remaining ambiguity in an already-locked scope.

**What changed from v5 (business decision layer added):** No architecture, technology, or prior decision changed — this pass adds the layer that turns the finished technical pipeline into an actual business recommendation, which was previously implied but not made explicit or mandatory. Additions: (1) a locked **Business Decision Workflow** (§4a) that turns threshold selection into a 12-step, evidence-driven process — including the explicit possibility that the model doesn't justify deployment; (2) a **Final Decision & Business Recommendation** output template (§4b) that the project must produce as an artifact, not just imply; (3) a **Business Case Study** (§12) as a separate stakeholder-facing deliverable from the README, with a locked 10-section structure — populated only after the real analysis is done, never pre-written; (4) an explicit **README vs. Case Study** audience/focus split (§13); (5) a **repository structure** (§14) reflecting the new deliverables. §6 and §11 are updated to make baseline comparison, sensitivity analysis, and the case study MUST HAVE, not implied extras. Week 8/9 of the timeline is restructured (business decision gets its own week) to make room for this without cutting anything else — the plan grows from 8 to 9 weeks for this reason alone.

---

## 0. What Changed and Why (read this before building)

| Area | v3 approach | Refinement | v4 approach |
|---|---|---|---|
| Project naming | Implied "AI Fraud Detection Platform" framing | The ML model + investigation are the actual product; an LLM narrative layer is a small supporting feature. Leading with "AI-powered" overstates the LLM's role and risks looking like AI-for-trend rather than AI-for-purpose. | Named **"Fraud Risk Analytics & Detection System"**, with **"Grounded GenAI Analyst Explanations"** as one labeled capability inside it. |
| Frontend | Next.js kept but flagged as "optional, not necessary" | A critic review classified Next.js as optional rather than bad — and since Vercel was already the deployment choice, a slightly more polished end-to-end artifact is worth the extra time, as long as it's kept small. | **Keep Next.js + Vercel.** Scope tightly: one main scoring/demo page + one small methodology/analytics page. Never becomes a frontend project. |
| LLM hosting | Anthropic/OpenAI API, used offline-only to avoid live cost | Even offline-only, this still relies on a paid API's free trial credits — not a genuinely permanent $0 solution, and adds an external dependency for something that only needs to run once. | **Local open-source model via Ollama** (e.g. Llama 3.2 3B, Phi-3-mini) for the one-time offline narrative generation. Zero cost, zero external dependency, fully reproducible on your own machine. |
| LLM value | SHAP → LLM narrative, presented as the AI differentiator | Restating structured SHAP output as prose is a real but modest UX layer on its own — mostly cosmetic unless something verifies the output stays grounded. | Add a **grounding validator**: a small script that checks the generated narrative doesn't introduce any feature name, number, or claim absent from the SHAP evidence it was given. This is what turns the LLM layer from decoration into a real (if small) piece of applied LLM engineering. |
| README claims | Generally accurate but some phrases invite overclaiming | "Real-time," "AI-powered," "production-ready," "monitoring," "zero-cost" all needed precise, defensible wording. | Locked wording table in §10 — use these exact phrasings, not the punchier-sounding alternatives. |
| Final scope | Scattered across Keep/Simplify/Removed lists | Needed one authoritative, final list. | §11 replaces the old scope checklist with a single locked MUST / SHOULD / OPTIONAL / REMOVE list. |

*(Everything from v2/v3 — neutral investigation framing, approximate client identity, honest split-strategy discovery, class-weighting over SMOTE, SHAP reason codes, Postgres as serving-layer-only, no PaySim/RAG/Kafka/Airflow/Kubernetes — carries forward unchanged.)*

---

## 1. Problem Framing

> **"Given historical transaction data, can we build a fraud-risk model that is honest about its evaluation methodology, interpretable at the individual-transaction level, deployable behind an API with a proper data layer, and observable after deployment — while explicitly investigating (not assuming) how entity overlap and temporal structure affect the reliability of different validation strategies?"**

The differentiation is not the model. It's the investigation, the honesty about what "real-time" and "explainable" actually mean, the fact that the data layer and monitoring are real, not decorative — and, finally, whether the whole pipeline answers the question a stakeholder actually asks: **"What did the analysis discover, what does it mean for the business, and what should the business actually do?"** A model with strong PR-AUC that never gets translated into an operating threshold and a plain-language recommendation is an unfinished analysis, not a finished one. §4a/§4b make this final translation step a required part of the project, not an afterthought.

---

## 2. Architecture Overview

```
IEEE-CIS raw data
      ↓
[Phase A: Data Integrity Investigation]   ← headline deliverable, own report
  - approximate client-ID reconstruction & overlap analysis
  - split-strategy decision (evidence-based, neutral framing)
  - V/D/C feature audit (keep/drop/investigate, documented)
      ↓
[Automated Data Quality Checks]   ← separate, repeatable, distinct from Phase A
  - schema / dtype / range / duplicate / label validity checks
      ↓
Feature Engineering (pandas: velocity, amount z-score, time-since-last, merchant freq)
  — each new feature checked against the Phase A audit for redundancy/leakage
  — raw + feature-engineered training data stays in local parquet/CSV only
      ↓
Modeling: Logistic Regression baseline → XGBoost/LightGBM (class-weighted)
      ↓
Evaluation: PR-AUC, recall@fixed-FPR, cost-matrix-based threshold selection
      ↓
SHAP → aggregated to top-5 reason codes per transaction
      ↓
Offline: generate templated narratives via a LOCAL LLM (Ollama, e.g. Llama 3.2
  3B / Phi-3-mini), temp 0, for held-out demo set ONCE — no paid API, no
  external dependency
      ↓
Grounding validator (ships with the LLM layer, not a separate add-on): checks
  each narrative introduces no feature name/number/claim absent from the SHAP
  evidence it was given — without this, the LLM layer is decoration
      ↓
PostgreSQL  ← SERVING/LOGGING LAYER ONLY: a small held-out demo replay slice,
               predictions, reason codes, stored narratives, model_runs
               (metrics per training run). Full raw/training data never
               loaded here — no reason to, adds infra with no use.
      ↓
FastAPI (/predict → score + reason codes; reads/writes Postgres)
      ↓
   ┌────────────────────────┴────────────────────────┐
   ↓                                                    ↓
Next.js (Vercel) demo — 2 pages MAX:              Power BI dashboard
 1) main scoring/demo page (live-scores            (connects to Postgres:
    held-out transactions via FastAPI,             fraud trends, PR/FPR,
    shows reason codes + stored narrative,         cost-saved, + Model
    labeled "simulated stream")                    Health monitoring tile)
 2) small methodology/analytics page
    (investigation summary, key metrics)
      ↓
[Business Decision Workflow]   ← §4a — threshold sweep, cost/FP-rate/review-
  volume per threshold, baseline comparison, sensitivity analysis. Runs on
  the deployed model's held-out predictions already stored in Postgres —
  no new infra, just analysis over data that already exists.
      ↓
Final Decision & Business Recommendation   ← §4b — one concrete artifact:
  recommended threshold + trade-off + operational recommendation, allowed
  to conclude "not worth deploying" if that's what the evidence shows
      ↓
Business Case Study   ← §12 — separate stakeholder-facing document,
  structure locked now, populated only after the real analysis is done
```

---

## 3. Data Sourcing

- **Primary and only dataset: IEEE-CIS Fraud Detection** (Kaggle). Real, messy, high-cardinality, large enough for genuine investigation.
- No PaySim, no second dataset. The investigation phase and the Postgres-backed replay of real held-out data replace the need for a separate "demo" data source.

---

## 4. Week-by-Week Plan (9 weeks, part-time alongside college)

### Week 1 — Data Integrity Investigation (headline deliverable) ✅ COMPLETE

**Confirmed findings (do not re-investigate; use these numbers in all downstream docs):**

- `TransactionDT` confirmed as a **relative delta in seconds** from an undisclosed origin. Range: `[86,400 – 15,811,131]`. Span: **182 days (26 weeks)**. Never reconstruct an absolute calendar date.
- Dataset: **590,540 rows × 434 columns** (post-merge). Fraud rate: **3.499%** (20,663 fraud). Identity join coverage: **23.8%** (identity features are sparse for most rows — handle missingness explicitly).
- Fraud rate rises slightly from first half (3.40%) to second half (3.61%) — confirming mild temporal drift.
- Approximate client proxy built from `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain` → **94,846 unique proxies**, avg 6.2 transactions/proxy.
- Entity overlap: **random split = 74.7%** (inflated; new entities have 32% lower fraud rate → metrics artificially optimistic), **temporal split = 67.6%** (realistic), **grouped = 0%** (too conservative, breaks temporal structure).
- **Chosen validation strategy: Temporal split at TransactionDT ≤ 12,192,854 (80th percentile).** Train: 472,432 rows. Test: 118,108 rows. Grouped entity split used as supplementary lower-bound benchmark only.
- `D1` = time-since-last-transaction (|r|=1.000 with our planned engineered feature). `C1` = velocity proxy (|r|=1.000). **Do not rebuild these in Week 3 — use D1 and C1 directly.**
- 162 near-duplicate V-feature pairs (|r| ≥ 0.98). 7 V-missingness clusters. Top predictors: V257, V201, V246, V200 (|r|=0.26–0.28 with fraud).
- 5 D-features show PSI > 0.10 under temporal split (D4, D6, D10, D14, D15) — structural drift, flag for monitoring. Not a reason to drop them.
- **Deliverable:** `docs/01_data_integrity_investigation.md` ✅ (populated with real numbers).

### Week 2 — Data Layer + Automated Quality Checks ✅ COMPLETE

- Stood up PostgreSQL (Supabase free tier) schema (`sql/schema.sql`) as a **serving/logging layer only**: DDL for `predictions`, `model_runs`, and `demo_replay` tables with optimized indexes (partial index for high-risk manual review queue).
- Enforced storage segregation boundary: Full raw and merged training data (590,540 rows × 434 cols) remains in local columnar Parquet (`data/processed/train_merged.parquet`, 84.1 MB) to prevent exhausting Supabase free-tier limits (<500MB).
- Built automated data quality validation engine (`src/validation/data_quality.py`) and 20-test pytest suite (`tests/test_data_quality.py`): schema presence, memory dtypes, `isFraud ∈ {0, 1}` target integrity, `TransactionID` uniqueness, physical/domain range bounds, critical 0% nulls, and temporal partition verification (20/20 tests passed).
- Built held-out demo slice extractor (`src/data/make_demo_slice.py`): generates 1,500 test transactions from `TransactionDT > 12,192,854` (517 KB Parquet / 418 KB JSON seed).
- Authored BI analytics queries (`sql/analytics_queries.sql`) for Power BI and monitoring.
- **Deliverables:** `src/validation/data_quality.py` ✅, `tests/test_data_quality.py` ✅, `sql/schema.sql` ✅, `sql/analytics_queries.sql` ✅, `src/data/make_demo_slice.py` ✅, `docs/02_data_quality_notes.md` ✅.

### Week 3 — Feature Engineering ✅ COMPLETE

**Scope implemented based on Week 1 audit findings:**

| Feature | Action | Reason |
|---|---|---|
| Transaction velocity (rolling count by card) | ❌ **Skipped — used C1 directly** | C1 is identical (\|r\|=1.000 with planned feature) |
| Time since last transaction | ❌ **Skipped — used D1 directly** | D1 is identical (\|r\|=1.000). D2 is also highly correlated (\|r\|=0.973). |
| Amount z-score by card/merchant (`amt_zscore_card1`, `amt_zscore_card1_addr1`, `amt_zscore_email`) | ✅ **Built** | High-value deviation signal; $|r| \le 0.03$ with C1/D1 (non-redundant) |
| Log-transformed TransactionAmt (`log_TransactionAmt`) | ✅ **Built** | Compresses heavy right-tail skewness; $|r| \le 0.03$ with C1/D1 |
| Frequency encodings (`freq_card1`, `freq_card2`, `freq_addr1`, `freq_ProductCD`, `freq_R_emaildomain`, etc.) | ✅ **Built & audited** | Strong non-linear predictive signal (`freq_R_emaildomain` $r = +0.156$, `freq_addr2` $r = -0.158$) |
| Cyclical hour-of-day features (`hour_sin`, `hour_cos`) | ✅ **Built** | Captures 24-hour diurnal cycle |
| Cyclical day-of-week features (`dow_sin`, `dow_cos`) | ✅ **Built** | Captures 7-day weekly cycle |
| Email consistency flag (`email_match_flag`) | ✅ **Built** | Consistency between payer and recipient domains ($r = +0.148$) |

- **Strict Leakage Prevention:** All frequency mappings and amount group statistics fit strictly on `TransactionDT ≤ 12,192,854` (Train).
- **Cross-Correlation Audit:** Verified that all 24 newly engineered features maintain $|r| < 0.85$ with $C1$ and $D1$ (maximum $|r|$ observed was 0.243), proving genuine additive predictive value.
- **Deliverables:** `src/features/engineer.py` ✅, `src/features/build_features.py` ✅, `tests/test_feature_engineering.py` (8/8 passed) ✅, `notebooks/03_feature_engineering.ipynb` ✅, `data/processed/train_features.parquet` (74.9 MB) ✅, `data/processed/test_features.parquet` (19.6 MB) ✅, `models/feature_pipeline.joblib` ✅, `data/processed/feature_metadata.json` ✅.

### Week 4 — EDA / Storytelling ✅ COMPLETE

- **5 Statistically-Grounded Stakeholder Stories** computed strictly on the training partition ($N=472,432$, $TransactionDT \le 12,192,854$):
  1. **Diurnal Attack Window:** Night off-peak hours (00:00–06:59) exhibit a **1.362x risk multiplier** (95% CI: 1.310x – 1.416x) vs. daytime hours.
  2. **The Self-Transfer Recipient Anomaly:** In digital delivery/remittance flows with a recipient email, identical purchaser and recipient domains ($P == R$) produce a **9.29% fraud rate** vs. **2.82%** for genuine cross-transfers (**3.296x risk ratio**, 95% CI: 3.031x – 3.584x) and 2.00% for standard retail (4.64x multiplier).
  3. **Card-Level Relative Deviations:** Transactions exceeding $3.0\sigma$ above their card's historical baseline have a **4.422% fraud rate** vs **3.359%** at baseline (**1.317x relative risk**, 95% CI: 1.190x – 1.456x).
  4. **Product Category & Channel Disparities:** Category 'C' concentrates the highest fraud density (**11.69%**), whereas category 'W' generates **> 65% of total net fraud dollar losses**.
  5. **The Identity Capture Paradox:** Transactions with attached identity metadata have a **7.55% fraud rate vs. 2.13%** for unverified flows (**3.545x risk ratio**, 95% CI: 3.440x – 3.652x), proving identity capture is an adversarial risk indicator.
- **Deliverables:** `src/eda/insights.py` ✅, `src/eda/__init__.py` ✅, `tests/test_eda_insights.py` (7/7 passed) ✅, `notebooks/02_eda_and_storytelling.ipynb` ✅, `data/processed/eda_insights_summary.json` ✅.

### Week 5 — Modeling ✅ COMPLETE
- **Logistic Regression Baseline:** Trained on scaled/imputed numerical and frequency features with `class_weight="balanced"`. Test PR-AUC: `0.2746`, ROC-AUC: `0.8092`, Recall @ 1% FPR: `15.08%`, Recall @ 5% FPR: `41.76%`.
- **Champion LightGBM Classifier:** High-capacity gradient boosting with `scale_pos_weight=27.46`. Test PR-AUC: `0.5441` (+98.1% relative gain over baseline), ROC-AUC: `0.9035`, Recall @ 1% FPR: `46.63%` (3.09x capture improvement), Recall @ 5% FPR: `65.95%` (+24.19 pp).
- **1,000-Resample Non-Parametric Bootstrap 95% Confidence Intervals:**
  - PR-AUC: `[0.5282 – 0.5607]`
  - ROC-AUC: `[0.8982 – 0.9087]`
  - Recall @ 1% FPR: `[44.90% – 48.24%]`
  - Recall @ 5% FPR: `[64.46% – 67.45%]`
- **Ablation & Generalization Audit:** Unweighted LightGBM PR-AUC: `0.5556`; Unseen entity lower-bound benchmark ($N=10,952$, 0% entity overlap): PR-AUC `0.4487`, Recall @ 1% FPR `36.36%` (~17.5% expected performance decay on novel entities).
- **Deliverables:** `src/models/evaluation.py` ✅, `src/models/train.py` ✅, `tests/test_models.py` (5/5 passed; 40/40 total repository tests passed) ✅, `notebooks/04_model_training_and_evaluation.ipynb` ✅, `models/champion_model.joblib` ✅, `models/baseline_logistic_regression.joblib` ✅, `models/model_metrics.json` ✅.

### Week 6 — Explainability + Offline GenAI Narratives ✅ COMPLETE
- **SHAP Reason Code & Collinearity Consolidation Engine (`src/explainability/reason_codes.py`)**: Consolidated 162 near-duplicate collinear $V$-feature pairs ($|r| \ge 0.98$) into coherent driver clusters (e.g. *Payment Activity Volume Cluster* for $V95/V101/V279/V293$), mapped features to human-readable domain descriptors, and integrated predefined business policy actions (`APPROVE`, `STEP_UP_AUTH`, `MANUAL_REVIEW`).
- **TreeSHAP Explainer (`src/explainability/shap_explainer.py`)**: Implemented interactive inference-time and batch TreeSHAP feature attributions on Champion LightGBM.
- **Multi-Provider GenAI Narrative Layer (`src/explainability/llm_client.py` & `src/explainability/narrative_generator.py`)**: Strict $0 hierarchy: Ollama (Primary local model, temp 0) $\rightarrow$ Deterministic Template (Baseline guarantee) $\rightarrow$ Grok/xAI API (Optional experiment). Generates structured 4-section analyst assessments with automatic fallback-on-rejection.
- **Grounding Validator Engine (`src/validation/grounding_validator.py`)**: Automated verification engine distinguishing direct facts, derived claims (recalculated against baselines), feature existence, directional consistency (+/- SHAP), and anti-speculation rules.
- **Offline Batch Generation Pipeline (`src/explainability/batch_generate_narratives.py`)**: Enriched the 1,500 held-out test transactions in `data/processed/demo_replay_slice.parquet` and `data/processed/demo_replay_slice.json`. Generated empirical audit manifest `data/processed/grounding_validation_report.json` (93.47% empirical initial pass rate, 100.0% final verified rate after fallback substitution).
- **Unit Test Suites (`tests/test_explainability.py` & `tests/test_grounding_validator.py`)**: 12 new pytest unit tests (52/52 total repository tests passing).
- **Visual Storytelling Notebook (`notebooks/05_shap_and_narrative_generation.ipynb`)**: Global beeswarm plots, collinearity consolidation demos, case studies, and grounding audit checks.
- **Deliverables:** `src/explainability/reason_codes.py` ✅, `src/explainability/shap_explainer.py` ✅, `src/explainability/llm_client.py` ✅, `src/explainability/narrative_generator.py` ✅, `src/validation/grounding_validator.py` ✅, `src/explainability/batch_generate_narratives.py` ✅, `tests/test_explainability.py` (6/6 passed) ✅, `tests/test_grounding_validator.py` (6/6 passed; 52/52 total repository tests passed) ✅, `notebooks/05_shap_and_narrative_generation.ipynb` ✅, `data/processed/demo_replay_slice.parquet` ✅, `data/processed/demo_replay_slice.json` ✅, `data/processed/grounding_validation_report.json` ✅.

### Week 7 — Deployment (data layer + API + monitoring) ✅ COMPLETE
- **FastAPI Serving Engine (`api/main.py`, `api/routes.py`, `api/schemas.py`, `api/db.py`, `api/config.py`)**:
  - Live `/predict` & `/predict/batch` endpoints: loads Champion LightGBM booster, fitted feature pipeline, and TreeSHAP explainer with reason code aggregation. Measures and returns empirical request latency (`latency_ms`) on every prediction.
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
- **Deliverables:** `api/main.py` ✅, `api/routes.py` ✅, `api/schemas.py` ✅, `api/db.py` ✅, `api/config.py` ✅, `src/data/seed_supabase.py` ✅, `dashboard/powerbi_setup_guide.md` ✅, `dashboard/dax_measures_and_model_health.md` ✅, `dashboard/export_analytics_extracts.py` ✅, `dashboard/data/` CSV extracts ✅, `render.yaml` ✅, `Dockerfile` ✅, `.env.example` ✅, `docs/04_render_deployment_guide.md` ✅, `tests/test_api.py` (14/14 passed; 66/66 total tests passed) ✅.

### Week 8 — Business Decision: Threshold, Cost, and Sensitivity Analysis
- Run the full 12-step workflow in §4a end to end on the deployed model's held-out predictions (already logged in Postgres from Week 7 — no new data pipeline needed).
- Sweep a realistic range of thresholds; at each, compute fraud capture, false-positive rate, expected manual-review volume, and expected financial cost, using cost assumptions stated explicitly (and sourced/justified, not invented).
- Compare the recommended operating point against the naive baseline (Logistic Regression at default threshold, or "flag everything above raw amount X") to answer: *is the model actually worth the added complexity?* — a negative answer is an acceptable outcome, not a failure of the project.
- Run the sensitivity/scenario analysis (§4a step 9–10): re-derive the recommendation under fraud-cost ±20%, false-positive-cost changes, and review-capacity 5% vs 10%. State plainly whether the recommended threshold survives these changes or whether the "right" threshold is itself capacity/assumption-dependent — either finding is reportable.
- **Deliverable:** `03_business_decision_and_threshold_analysis.md` containing the completed decision summary from §4b template (with real numbers, not placeholders) — this becomes the primary source for §12's Case Study §07–09.

### Week 9 — Frontend, Business Case Study, Polish
- Next.js on Vercel — **two pages max**: (1) main scoring/demo page replaying held-out transactions through the live FastAPI endpoint, showing score + reason codes + stored narrative, explicitly labeled "simulated stream over real held-out data"; (2) a small methodology/analytics page summarizing the investigation finding and key metrics. Do not let this expand beyond two pages — no auth, no routing complexity, no extra views.
- Write the **README** (developer/technical audience — see §13): Problem → Investigation findings → Data quality approach → Approach → Key decisions & tradeoffs → Results → Business decision summary (link to full case study) → Monitoring → Limitations → What you'd do with more time/data.
- Write the **Business Case Study** (stakeholder audience — see §12/§13), using the real results from Weeks 1–8. Populate every placeholder in §12's structure with actual findings — never ship it with example numbers.
- 3–4 min walkthrough video. README with architecture diagram, exact wording from §10 for every claim, live demo link, Power BI screenshot, link to the case study.

---

## 4a. Business Decision Workflow (How the Final Threshold Is Determined)

This is the step that turns a trained model into an actual recommendation. It runs entirely on artifacts that already exist by Week 7 (held-out predictions logged in Postgres) — no new data collection, no new infrastructure.

**The recommendation is not decided in advance.** The workflow below must be run for real, on real numbers, and it must be allowed to conclude that the model doesn't justify deployment over the baseline — that is a valid, reportable outcome, not a failure state.

1. Establish realistic business cost assumptions (average loss per missed fraud, average cost of a manual review, stated explicitly and sourced or clearly labeled as an assumption where no public figure exists).
2. Establish operational constraints — realistic manual-review capacity (e.g. as a % of daily transaction volume a small fraud team could plausibly review).
3. Evaluate the model across a range of thresholds (not just one default cutoff).
4. Calculate fraud capture (recall) at each threshold.
5. Calculate false-positive rate at each threshold.
6. Calculate expected manual-review volume at each threshold.
7. Estimate expected financial cost at each threshold, combining missed-fraud cost and review cost.
8. Compare the best candidate threshold's cost against the baseline (naive rule or default-threshold Logistic Regression) — quantify the improvement, or the lack of one.
9. Run sensitivity/scenario analysis: fraud cost ±20%, false-positive cost changes, review capacity at 5% vs. 10%, and at least one alternate threshold choice.
10. Determine whether the recommendation is robust to reasonable changes in these assumptions — if the "right" threshold changes with capacity or cost assumptions, that dependency is itself a finding, not a flaw to hide.
11. Select the recommended operating point (or state that no threshold clears the bar over baseline, if that's what the evidence shows).
12. Translate the result into a plain-language stakeholder recommendation (§4b), including what would change it.

**Do not assume a single universally optimal threshold exists.** If the appropriate choice depends on review capacity or cost assumptions, report that as a finding rather than picking one number and hiding the dependency.

---

## 4b. Final Decision & Business Recommendation (Required Output)

The project must produce this as an actual artifact (feeds directly into Week 8's deliverable and the Case Study's §07–09) — not end with a bare model-performance number. The values below are illustrative placeholders only; the shipped version must contain the project's real results.

**Decision summary must contain:**
1. Key analytical findings (from the investigation and modeling, in plain language)
2. Recommended threshold
3. Fraud capture at that threshold
4. False-positive rate at that threshold
5. Expected manual-review volume
6. Estimated financial impact, presented as a range with stated assumptions
7. Comparison against the baseline (quantified — how much better, if at all)
8. Operational recommendation — how the model should actually be used
9. Major assumptions the recommendation depends on
10. Limitations
11. Monitoring requirements going forward
12. Conditions that would trigger re-evaluation of the threshold or the model itself

*Illustrative format only — replace every value with real analysis results:*

> **Recommended threshold:** X
> At this threshold: Fraud captured X% · False-positive rate X% · Transactions requiring review X% · Estimated expected cost ₹X–₹Y (vs. ₹A–₹B under baseline)
>
> **Recommendation:** "Use the model as a risk-ranking/manual-review-prioritization system rather than an automatic-decline system, because [reason grounded in the actual precision/recall trade-off found]."

**The recommendation must not default to automatic transaction blocking.** A reasonable structure — high risk → manual review/additional verification, medium risk → additional checks, low risk → normal processing — is one example, not a template to fill in without checking whether it's actually what the numbers support. The analysis decides the structure, not the other way around.

---

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy |
| Investigation | pandas groupby/overlap analysis, correlation analysis |
| Data quality | pytest-based checks (schema, dtype, range, duplicates) |
| Database | PostgreSQL on **Supabase** free tier — serving/logging layer only: predictions, model_runs, demo-replay slice (see §9) |
| Modeling | scikit-learn, XGBoost/LightGBM |
| Explainability | SHAP (aggregated to reason codes) |
| AI/LLM layer | **Local model via Ollama** (e.g. Llama 3.2 3B, Phi-3-mini) — used **offline only**, one-time, to pre-generate stored narratives; no paid API |
| Grounding validator | Small Python script checking generated narratives against SHAP evidence — ships with the LLM layer, not separately optional |
| Backend/API | FastAPI |
| Demo frontend | Next.js, deployed on Vercel — **two pages max** (scoring demo + methodology/analytics) |
| BI layer | Power BI, connected directly to Postgres |
| Monitoring | Logged predictions in Postgres + Power BI Model Health tile |
| Deployment | Render free tier (API), Vercel free tier (frontend) |
| Versioning/CI | GitHub, GitHub Actions (free for public repos) |

---

## 6. Scope Checklist

*(Superseded by the locked final scope in §11 — kept here for traceability of what changed across revisions.)*

**Keep (non-negotiable):**
- [x] Data Integrity Investigation as a standalone, documented deliverable (Week 1)
- [x] Evidence-based split strategy (not assumed) (Week 1)
- [x] Feature audit of `V`/`D`/`C` columns with documented reasoning (Week 1)
- [x] Class-weighted XGBoost/LightGBM + LR baseline (Week 5)
- [ ] Cost-matrix-based threshold selection
- [x] SHAP reason codes (Week 6)
- [x] PostgreSQL as the real serving/logging system of record — predictions, model_runs, demo slice (not the full raw dataset, not CSV) (Week 2)
- [x] Automated data-quality checks (separate from the investigation) (Week 2)
- [ ] One deployed FastAPI endpoint + Next.js demo (two pages max) on real held-out data
- [ ] Basic post-deployment monitoring surfaced in Power BI
- [ ] Business Decision Workflow (§4a) — threshold sweep, quantified baseline comparison, sensitivity analysis — run for real, allowed to conclude the model isn't worth deploying
- [ ] Final Decision & Business Recommendation artifact (§4b) with real numbers, not a bare model-metric conclusion
- [ ] Business Case Study (§12) as a stakeholder-facing deliverable, separate from the README, populated only after the analysis is complete

**Simplify:**
- [x] LLM layer → local model (Ollama), offline-generated, template-constrained, stored not live (Week 6)
- [ ] Next.js frontend → two pages max, no auth/routing complexity
- [ ] Power BI → focused pages (analytics + Model Health), clearly labeled

**Removed:**
- [ ] PaySim
- [ ] RAG chatbot
- [ ] Live per-request LLM calls in the public demo
- [ ] Paid LLM API entirely (moved to local model, §0)
- [ ] Kubernetes/Kafka/Airflow/Spark/Terraform/complex CI-CD — not appropriate at this scope

---

## 7. Resume Bullet Points (draft)

- Investigated how entity overlap and temporal structure affect validation reliability in a real-world fraud dataset (IEEE-CIS) — reconstructed approximate client identity from pseudo-ID columns, measured overlap across candidate train/test splits, and selected a validation strategy based on measured evidence rather than default assumptions.
- Audited pre-engineered feature blocks against hand-built features and target correlation; investigated potential leakage and selected a validation strategy based on measured entity overlap and feature behavior. *(Update with your actual finding once complete.)*
- Built a class-weighted XGBoost fraud model with cost-matrix-driven threshold selection, achieving [X]% recall (95% CI: [X_low]–[X_high]) at [Y]% false-positive rate on the held-out set; explained individual predictions via SHAP-derived reason codes.
- Designed a PostgreSQL serving/logging layer (predictions, model runs, demo-replay data — deliberately scoped to exclude the full raw dataset), with automated data-quality checks and basic post-deployment monitoring surfaced in Power BI.
- Built a grounded GenAI explanation layer using a locally-hosted open-source LLM (Ollama) constrained to restate SHAP evidence only, with an automated validator checking generated text against the underlying evidence.
- Deployed the model behind a FastAPI endpoint with a Next.js demo replaying real held-out transactions, and a Power BI dashboard summarizing fraud trends, model performance, and system health for non-technical stakeholders.
- Ran a threshold-selection and cost-sensitivity analysis comparing the model against a naive baseline across a range of manual-review-capacity and cost-assumption scenarios; translated the result into a plain-language operating recommendation for a non-technical stakeholder audience. *(Update with the actual recommendation — including "not worth deploying" — once complete.)*

---

## 8. Time-box Notes

- If time gets tight, protect (in order): Data Integrity Investigation → cost-matrix threshold selection → Business Decision Workflow & Final Recommendation (§4a/§4b) → the Postgres data layer → automated data-quality checks. These are what differentiate this from a generic IEEE-CIS notebook — the business decision step is *not* the first thing to cut; it's what the rest of the project is for.
- The Next.js frontend and Power BI polish are the first things to trim to a minimal version — two working pages and a two-tile dashboard are enough; don't let frontend polish eat into the write-up or the business decision analysis.
- The LLM narrative layer and calibration are genuinely optional — cut these first if the clock runs out, not the investigation, the modeling, or the business decision workflow. **If you cut, cut the whole LLM layer together with its validator — never keep the narratives and drop the validator; an unvalidated LLM layer is worse than no LLM layer.**
- Don't skip the write-up. The investigation phase is only valuable if it's documented well enough to talk through in an interview without re-deriving it live — the same applies to the business decision: an interviewer will ask "why that threshold," and the answer needs to already be written down.

---

## 9. Keeping the Entire Stack at $0

| Component | Free option | Notes |
|---|---|---|
| Dataset | Kaggle (IEEE-CIS) | Free, requires free Kaggle account. |
| Database | **Supabase** free tier (Postgres) | Chosen over Render's free Postgres, which auto-expires after 90 days. Supabase free-tier projects pause after ~1 week of inactivity and need manual reactivation — mention this in the README, or set up a trivial scheduled ping to keep it warm. |
| API hosting | **Render** free web service | Cold-starts after inactivity — acceptable for a portfolio demo; mention it in the README so it's not mistaken for a bug. |
| Frontend hosting | **Vercel** Hobby tier | Free for non-commercial use (a portfolio project qualifies); no practical time limit for low traffic, but capped bandwidth/build minutes. |
| BI tool | **Power BI Desktop** (free) + **Publish to Web** | Power BI Pro isn't required for a personal portfolio piece; "Publish to Web" is free but makes the report publicly viewable — fine for a portfolio, just don't put sensitive data in it. |
| LLM narratives | **Local model via Ollama**, run on your own machine | Genuinely $0, no external dependency, no expiring free tier — this is the most defensible "free" claim in the whole stack, since it never touches a paid API even during development. |
| CI | GitHub Actions | Free for public repositories. |
| Source/docs | GitHub | Free for public repositories. |

**Accurate claim to use in the README:** *"Built entirely on free tiers of hosted services, each with its own usage limits and inactivity behavior — no ongoing service costs."* Not an unconditional "zero-cost forever" — say so precisely, it's more credible than the punchier version.

---

## 10. Precise Wording for Every Claim (use these exact phrasings)

| Tempting phrase | Honest, still-impressive version |
|---|---|
| "AI Fraud Detection Platform" | **"Fraud Risk Analytics & Detection System"** with "Grounded GenAI Analyst Explanations" as a labeled sub-feature |
| "Real-time fraud detection" | "Simulated real-time inference — replays held-out transactions through the deployed model" |
| "Production-ready system" | "Deployment-ready portfolio demo" |
| "AI-powered explanations" | "Model explanations generated offline by a local LLM, grounded strictly to SHAP evidence" |
| "Explainable AI" | "SHAP-based feature attribution, aggregated into reason codes" (explains the model's logic, not the true causal reason for fraud) |
| "Fraud prevention system" | "Fraud risk detection/scoring system" (it flags risk; it doesn't itself prevent anything without a downstream action) |
| "Client identity reconstruction" | "Approximate, correlation-based entity identifiers (not a confirmed ground-truth key)" |
| "Completely free / zero-cost" | "Built entirely on free tiers of hosted services, each with its own usage limits and inactivity behavior" |
| "Model monitoring" | "Basic observability metrics (volume, score distribution, precision/recall when labels are available)" — not a production MLOps monitoring system |
| "Deployed ML system" | "Portfolio-scale deployed demo" — not something serving real customer traffic |

---

## 11. Final Locked Scope

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
- Statistical confidence framing wherever a business claim is made — not just EDA: bare point estimates aren't enough. Covers (a) EDA insights, (b) the headline recall@FPR metric (report a bootstrapped or Wilson confidence interval alongside the point estimate, not just the number), and (c) the Power BI cost-saved estimate (state the assumptions the estimate depends on and a plausible range, not a single dollar figure)

**OPTIONAL**
- Local LLM (Ollama) grounded narrative layer — **if built, the grounding validator below is not separately optional; it ships as part of the same unit.** Do not deploy narrative generation without it.
- Grounding validator for the LLM output — conditionally required (see above); only skippable by skipping the LLM layer entirely
- GitHub Actions CI
- Calibration analysis (reliability diagram)

**REMOVE**
- RAG chatbot
- PaySim
- Kafka, Airflow, Kubernetes
- Unused Supabase features (auth, storage, realtime)
- Any live, paid LLM API call — including for one-time offline generation

---

## 12. Business Case Study (Separate Stakeholder-Facing Deliverable)

The case study is a distinct artifact from the README (see §13) — it answers "what problem did we have → what did we investigate → what did we discover → what did we analyze → what decision should be made → why does it matter?" for a non-ML audience. It is **not** another technical walkthrough.

**Build the structure now. Populate it only after Week 8's real analysis is complete.** Never ship placeholder numbers in the final version — every `[X]`/`[Y]` below must be replaced with the actual finding.

Target length: roughly 8–12 pages, or an equivalent web presentation.

**01 — Executive Summary** — one page: business problem, approach, major finding, recommended action, business impact. Understandable without ML background.

**02 — Business Problem** — why fraud detection matters here, who the stakeholder is, what decision they need to make, what operational constraints exist, what question the analysis answers.

**03 — Data Investigation** — dataset characteristics, missingness, class imbalance, temporal structure, entity overlap, suspicious features, leakage investigation. Centered on: *what did we discover that changed our methodology?*

**04 — Modeling Approach** — baseline, main model, feature engineering, class weighting, validation strategy, and why these choices were made. Not a tutorial — a decisions log.

**05 — Model Results** — PR-AUC, precision/recall, recall@fixed-FPR (with its confidence interval, per §11), confusion matrix, calibration if used, and a direct baseline comparison answering: *is the model actually better than the baseline?*

**06 — Explainability** — one real transaction example: fraud probability, top SHAP reasons, the human-readable explanation, and — if the LLM layer is used — a plain-language description of the grounding safeguard.

**07 — Business Impact & Threshold Decision** — one of the most important sections. Threshold comparison, fraud captured, false positives, review volume, expected cost, financial impact, then a stated **Recommendation:** "Use threshold X because..." sourced directly from §4b's completed output.

**08 — Sensitivity Analysis** — what happens as fraud cost, false-positive cost, and review capacity change; answers whether the recommendation is robust.

**09 — Final Business Recommendation** — explicitly: what the business should do, how the model should be used, what different risk levels should trigger, and why this is the recommended approach (not automatic blocking by default — see §4b).

**10 — Limitations & Next Steps** — IEEE-CIS limitations, cost-assumption limitations, label limitations, approximate entity reconstruction, differences from real production banking data, free-tier infrastructure limitations, and what a real fintech deployment would require next.

---

## 13. README vs. Case Study

Two deliverables, two audiences — do not duplicate one inside the other.

| | **README** | **Business Case Study** |
|---|---|---|
| Audience | Developers, technical interviewers, engineers | Recruiters, hiring managers, data analysts/scientists, business stakeholders |
| Focus | Architecture, code, setup, methodology, ML internals, API, database, deployment, testing, monitoring | Problem, investigation, findings, model value, business impact, decision, recommendation, limitations |
| Depth on ML internals | Full | Summarized — enough to trust the conclusion, not enough to reproduce the model |
| Depth on business decision | Linked, summarized | Full — this is the case study's centerpiece (§12 §07–09) |

---

## 14. Repository Structure

```
fraud-risk-analytics/
│
├── README.md
├── case-study/
│   └── fraud-risk-case-study.pdf        ← §12, populated after Week 8
│
├── notebooks/                            ← investigation, EDA, modeling
├── src/                                  ← feature engineering, training, evaluation
├── api/                                  ← FastAPI backend
├── frontend/                             ← Next.js demo (2 pages max)
├── sql/                                  ← analytics queries against predictions/model_runs
├── dashboard/                            ← Power BI file/export
├── tests/                                ← pytest data-quality + grounding-validator tests
├── models/                               ← saved model artifacts, versioned
└── docs/                                 ← 01_data_integrity_investigation.md,
                                             data_quality_checks notes,
                                             03_business_decision_and_threshold_analysis.md
```

Publishing the case study as a web page is a nice-to-have, not a requirement — the PDF/markdown in `case-study/` is sufficient on its own.

