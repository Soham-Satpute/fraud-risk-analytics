# Data Quality & Serving Architecture Notes
## Week 2 Engineering Deliverable

> **Core Philosophy:** *Investigation builds initial trust; automated data quality checks maintain it across every subsequent batch and deployment run.*

---

## 1. Paradigm Shift: Week 1 Investigation vs. Week 2 Data Quality

A common mistake in ML systems is conflating **exploratory data investigation** with **operational data quality validation**. In this project, they serve two distinct, non-overlapping functions:

```
┌────────────────────────────────────────────────────────┐
│ Week 1: Forensic Investigation (One-Time Trust)        │
│ ──────────────────────────────────────────────         │
│ • Entity proxy reconstruction (94,846 proxies)         │
│ • Leakage & overlap analysis (74.7% vs 67.6%)          │
│ • V/D/C collinearity & redundant feature audit         │
│ • Scientific selection of temporal validation split    │
└───────────────────────────┬────────────────────────────┘
                            │ Establishes Ground Truth & Invariants
                            ▼
┌────────────────────────────────────────────────────────┐
│ Week 2: Automated Quality Engine (Continuous Contract) │
│ ────────────────────────────────────────────────────── │
│ • Repeatable batch-level schema & dtype validation     │
│ • Target label integrity & binary encoding invariant   │
│ • Primary key uniqueness assertion                     │
│ • Physical & empirical numerical range bounds          │
│ • Critical feature 0% null threshold enforcement       │
│ • PostgreSQL serving & audit logging architecture      │
└────────────────────────────────────────────────────────┘
```

| Dimension | Week 1: Data Integrity Investigation | Week 2: Automated Data Quality Engine |
|---|---|---|
| **Cadence** | One-time forensic audit | Continuous, executed per data batch and CI run |
| **Objective** | Discover unknown dataset properties, leakage risks, and split behavior | Enforce known invariants and catch data corruption before training/serving |
| **Tooling** | Custom exploratory scripts, correlation matrices, PSI calculations | Modular Python library (`src/validation/data_quality.py`) + `pytest` suite |
| **Outcome** | Locked facts and methodological choices (`docs/01_data_integrity_investigation.md`) | Automated test reports, schema contracts, and serving DDL (`sql/schema.sql`) |

---

## 2. Storage Segregation Rationale ($0 Free-Tier Architecture)

### 2.1 Why Raw Data Stays Out of PostgreSQL

The raw IEEE-CIS dataset spans **590,540 rows × 434 columns** (~683 MB uncompressed CSV per table, ~2.5 GB in-memory representation, 84.1 MB compressed Parquet).

We enforce a strict architectural boundary: **The full raw and merged training datasets are NEVER loaded into PostgreSQL.**

**Key Rationale:**
1. **Free-Tier Storage Caps (Supabase 500 MB limit):** Loading 590k wide rows with indexes into Postgres would instantly consume the free storage quota, leaving no room for operational logs and risking database suspension.
2. **Read/Write Inefficiency:** ML model training routines (`XGBoost`, `LightGBM`, `scikit-learn`) require rapid columnar vector reads. Querying 434 columns over row-oriented Postgres over a network connection is orders of magnitude slower than local columnar Parquet reads (`pyarrow`/`pandas`).
3. **Separation of Concerns:** Relational databases are optimized for transactional integrity (OLTP) and operational indexing, not bulk analytical feature matrices.

### 2.2 PostgreSQL as the Serving & Logging System of Record

PostgreSQL on Supabase (Free Tier) is strictly reserved as an **operational serving, logging, and demo replay layer**:

```
                         ┌─────────────────────────────────┐
                         │ Local Columnar Parquet Storage  │
                         │ (data/processed/train_merged)   │
                         └────────────────┬────────────────┘
                                          │ Batch Training
                                          ▼
┌───────────────────────┐         ┌───────────────┐
│ Held-Out Demo Stream  │ ──────> │ FastAPI / ML  │
│ (demo_replay table)   │         │ Model Service │
└───────────────────────┘         └───────┬───────┘
                                          │ Logs Predictions
                                          ▼
                         ┌─────────────────────────────────┐
                         │ PostgreSQL on Supabase          │
                         │ (predictions & model_runs logs) │
                         └────────────────┬────────────────┘
                                          │ Analytical Queries
                                          ▼
                         ┌─────────────────────────────────┐
                         │ Power BI Dashboard / Metrics    │
                         └─────────────────────────────────┘
```

The database schema (`sql/schema.sql`) maintains three compact tables:
1. **`predictions`**: Operational audit log capturing `transaction_id`, calibrated `fraud_probability`, `predicted_risk_tier`, `decision_action`, `top_reason_codes` (JSONB), and grounded LLM narrative.
2. **`model_runs`**: System of record for model releases, validation split methodology, operating threshold policies, PR-AUC, and financial cost-benefit estimates.
3. **`demo_replay`**: A curated, lightweight held-out test partition slice (~1,500 rows, **< 550 KB**) used by the Next.js frontend and FastAPI backend to simulate real-time transaction streaming.

---

## 3. Automated Data Quality Check Suite

The data quality framework (`src/validation/data_quality.py`) implements seven invariant checks:

### 3.1 Schema & Dtype Conformance (`check_schema`)
- Asserts that all mandatory features (`TransactionID`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `card1`, `C1`, `D1`) and target `isFraud` (on labeled sets) are present.
- Blocks downstream execution immediately if columns are dropped or renamed.

### 3.2 Primary Key Uniqueness (`check_uniqueness`)
- Enforces strict uniqueness on `TransactionID`. Duplicate transaction records in financial pipelines lead to double-counting and inflated risk exposures.

### 3.3 Target Label Integrity (`check_target_labels`)
- Verifies `isFraud ∈ {0, 1}` with **0% null tolerance**.
- Flags invalid class values (e.g. `-1`, `2`, `NaN`) and logs the batch fraud rate.

### 3.4 Critical Feature Completeness (`check_critical_nulls`)
- While identity features (`id_01`–`id_38`) have expected high missingness (~76% nulls), core transaction fields (`TransactionAmt`, `TransactionDT`, `card1`, `ProductCD`) must have **0% missing values**.

### 3.5 Numerical Range & Sanity Bounds (`check_numeric_ranges`)
- `TransactionAmt`: Must be strictly positive ($0.01 to $35,000). Catches zero or negative amounts that indicate ingestion errors.
- `TransactionDT`: Minimum delta must satisfy $\ge 86,400$ seconds (Day 1 baseline established in Week 1).
- `card1`: Must fall within standard card issuer BIN ranges ($1,000$ to $20,000$).
- `C1` and `D1`: Non-negative numerical constraints.

### 3.6 Categorical Domain Validity (`check_categorical_domains`)
- Validates discrete domain membership for `ProductCD` (`{'W', 'H', 'C', 'S', 'R'}`), `card4` (card networks), and `card6` (card types).
- Emits structured warnings (`WARN`) if novel categories emerge, enabling tracking of category drift without hard pipeline failure.

### 3.7 Temporal Monotonicity & Span (`check_temporal_span`)
- Verifies that batch timestamps span positive intervals and respect dataset baseline origins.

---

## 4. Test Suite Execution & CI Integration

The data quality suite is tested via `pytest` (`tests/test_data_quality.py`), covering:
- **Synthetic Corrupted Data Tests:** Ensures check functions actively catch negative amounts, duplicate IDs, invalid labels, missing columns, and extreme outliers.
- **Partition Boundary Verification:** Formally verifies the Week 1 temporal cutoff ($TransactionDT \le 12,192,854$ for 472,432 train rows and $> 12,192,854$ for 118,108 test rows).

### Running Quality Checks

**1. Automated Pytest Suite:**
```bash
python -m pytest tests/test_data_quality.py -v
```

**2. Direct CLI Validation on Data Batches:**
```bash
python -m src.validation.data_quality --data data/processed/train_merged.parquet --sample 50000
```

**3. Generating Curated Held-Out Demo Slice:**
```bash
python src/data/make_demo_slice.py --samples 1500 --fraud-ratio 0.15
```

---

## 5. Summary of Week 2 Artifacts

| File | Purpose |
|---|---|
| [`src/validation/data_quality.py`](file:///c:/Codes/fraud-risk-analytics/src/validation/data_quality.py) | Reusable data quality validation engine & CLI |
| [`tests/test_data_quality.py`](file:///c:/Codes/fraud-risk-analytics/tests/test_data_quality.py) | Pytest test suite covering synthetic corruptions and live parquet data |
| [`sql/schema.sql`](file:///c:/Codes/fraud-risk-analytics/sql/schema.sql) | DDL for PostgreSQL serving/logging layer (`predictions`, `model_runs`, `demo_replay`) |
| [`sql/analytics_queries.sql`](file:///c:/Codes/fraud-risk-analytics/sql/analytics_queries.sql) | Analytical queries for Power BI and operational monitoring |
| [`src/data/make_demo_slice.py`](file:///c:/Codes/fraud-risk-analytics/src/data/make_demo_slice.py) | Held-out demo slice extractor for simulated real-time stream (< 550 KB) |
| [`docs/02_data_quality_notes.md`](file:///c:/Codes/fraud-risk-analytics/docs/02_data_quality_notes.md) | This engineering document |
