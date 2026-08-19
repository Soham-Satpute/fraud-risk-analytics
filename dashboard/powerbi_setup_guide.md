# Power BI Setup & Data Architecture Guide

> **Project:** Fraud Risk Analytics & Detection System  
> **Target Audience:** Business Intelligence Engineers, Data Analysts, Stakeholders  
> **Source Plans:** `AGENTS.md` (§2, §10, §11) & `fintech-fraud-analytics-plan-v6.md` (§2, §9)

---

## 1. Architectural Overview & Connection Modes

Power BI serves as the business observability and stakeholder reporting layer for the deployed fraud risk system. It connects to the **PostgreSQL serving layer on Supabase** (or reads the generated static CSV extracts for offline development).

```
 ┌────────────────────────────────────────────────────────┐
 │            PostgreSQL (Supabase Free Tier)             │
 │  - predictions (inference logs & SHAP reason codes)    │
 │  - model_runs (benchmark metrics & threshold logs)     │
 │  - demo_replay (held-out test stream)                  │
 └───────────────────────────┬────────────────────────────┘
                             │ (DirectQuery or Import)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │                    Power BI Desktop                    │
 │                                                        │
 │  Page 1: Executive Fraud Risk & Financial Impact       │
 │   - Cost-Saved Range Model (dynamic scenario sliders)  │
 │   - Operational Alert Rate & Tier Breakdown            │
 │   - Top Attributed Fraud Drivers (SHAP Aggregations)   │
 │                                                        │
 │  Page 2: Model Health & Labeled Replay Observability   │
 │   - Model Health Tile (Scored Vol, High-Risk Queue)    │
 │   - Precision, Recall & FPR (on Held-Out Test Replay)  │
 │   - Score Distribution Deciles & Population Stability  │
 └────────────────────────────────────────────────────────┘
```

---

## 2. Option A: Live Supabase PostgreSQL Connection

### Step 1: Obtain Connection Credentials
1. Log in to your [Supabase Dashboard](https://app.supabase.com/).
2. Navigate to **Project Settings** → **Database**.
3. Under **Connection Parameters**, locate:
   - **Host:** `db.<project-ref>.supabase.co`
   - **Database:** `postgres`
   - **Port:** `5432` (or `6543` for connection pooling)
   - **User:** `postgres`
   - **Password:** Your database password

### Step 2: Connect Power BI Desktop
1. Open **Power BI Desktop**.
2. Click **Get Data** → **PostgreSQL database** → **Connect**.
3. Enter:
   - **Server:** `db.<project-ref>.supabase.co:5432`
   - **Database:** `postgres`
   - **Data Connectivity mode:** Choose **Import** (recommended for free tier to minimize query traffic) or **DirectQuery** (for real-time dashboard updates).
4. Enter database username and password when prompted.
5. In the Navigator pane, select the three serving tables:
   - `public.predictions`
   - `public.model_runs`
   - `public.demo_replay`
6. Click **Load**.

---

## 3. Option B: Offline Static CSV Fallback (Zero Cloud Setup)

If you are developing offline or grading without live Supabase cloud credentials, use the pre-generated static extracts located in `dashboard/data/`:

1. Run the extract script:
   ```powershell
   python dashboard/export_analytics_extracts.py
   ```
2. In Power BI Desktop, click **Get Data** → **Text/CSV**.
3. Load the following files from `dashboard/data/`:
   - `predictions_summary.csv`
   - `score_distribution.csv`
   - `operational_metrics.csv`
   - `evaluation_metrics.csv`
   - `high_risk_review_queue.csv`

---

## 4. Data Modeling & Relationships

In the Power BI **Model View**, configure relationships as follows:

| From Table | Column | To Table | Column | Cardinality | Cross Filter |
|---|---|---|---|---|---|
| `predictions` | `transaction_id` | `demo_replay` | `transaction_id` | 1:1 | Both |
| `predictions` | `model_version` | `model_runs` | `model_version` | Many:1 | Single |

---

## 5. Free-Tier Inactivity & Refresh Best Practices

- **Supabase Pausing:** Supabase free tier databases pause after 7 days of inactivity. If a query error occurs during refresh, log into Supabase to unpause the project.
- **Scheduled Refresh:** In Power BI Service (Pro or Publish to Web), set scheduled refresh to once or twice daily to keep within free-tier database compute quotas.
