# Render Free-Tier Deployment & Verification Guide

> **Project:** Fraud Risk Analytics & Detection System  
> **Topic:** FastAPI Serving Layer Deployment on Render Free Tier  
> **Source Plan:** `fintech-fraud-analytics-plan-v6.md` (§5, §9, Week 7) & `AGENTS.md` (§2, §4, §6)

---

## 1. Deployment Architecture

The backend API is containerized and hosted on the **Render Free Web Service** tier, providing a public HTTPS endpoint for the Next.js portfolio demo frontend and automated testing.

```
 ┌────────────────────────────────────────────────────────┐
 │                    Render Web Service                  │
 │                                                        │
 │  FastAPI Application (api/main.py)                     │
 │  - Preloads Champion LightGBM & TreeSHAP Explainer     │
 │  - /health (Health & latency benchmark)                │
 │  - /predict (Inference & SHAP reason codes)            │
 │  - /replay (Held-out test stream)                      │
 │  - /monitoring/* (Operational & evaluation metrics)    │
 └───────────────────────────┬────────────────────────────┘
                             │ (Async logging)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │            Supabase PostgreSQL (Free Tier)             │
 │  - predictions / model_runs / demo_replay              │
 └────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Render Deployment

### Option A: Deploy via Blueprint (`render.yaml`)
1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New** → **Blueprint**.
3. Connect your GitHub repository.
4. Render automatically parses `render.yaml` and provisions the `fraud-risk-api` web service.
5. In the service settings, add your `DATABASE_URL` environment variable pointing to your Supabase PostgreSQL database.

### Option B: Manual Web Service Setup
1. In Render, click **New** → **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Name:** `fraud-risk-analytics-api`
   - **Environment:** `Python 3`
   - **Region:** `Oregon (US West)` or closest region to your Supabase instance
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** `Free`
4. Under **Environment Variables**, add:
   - `ENVIRONMENT` = `production`
   - `CORS_ORIGINS` = `http://localhost:3000,https://*.vercel.app`
   - `DATABASE_URL` = `postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`

---

## 3. Post-Deployment Verification Checklist

Run these commands against your live Render service URL (replace `https://fraud-risk-api.onrender.com` with your assigned URL):

### 1. Health & Startup Verification
```bash
curl -i -X GET https://fraud-risk-api.onrender.com/health
```
**Expected Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "Fraud Risk Analytics & Detection System API",
  "version": "1.0.0",
  "environment": "production",
  "model_loaded": true,
  "pipeline_loaded": true,
  "explainer_loaded": true,
  "database_connected": true,
  "benchmark_inference_latency_ms": 45.2
}
```

### 2. Live Transaction Scoring Verification
```bash
curl -i -X POST https://fraud-risk-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 3459526,
    "TransactionDT": 12198174,
    "TransactionAmt": 117.0,
    "ProductCD": "W",
    "card1": 4436,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  }'
```
**Expected Response:**
- HTTP status `200 OK`
- `predicted_probability` bounded in $[0.0, 1.0]$
- `predicted_risk_tier` in `{"LOW", "MEDIUM", "HIGH"}`
- `decision_action` in `{"APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW"}`
- `top_reason_codes` containing 5 structured risk drivers with directional attributions.
- Response header `X-Process-Time-Ms` populated with measured latency.

### 3. Demo Replay Stream Verification
```bash
curl -i -X GET "https://fraud-risk-api.onrender.com/replay?limit=5"
```
**Expected Response:**
- Returns 5 transactions from the 1,500 held-out test records with pre-computed narratives.

### 4. Operational Observability Verification
```bash
curl -i -X GET https://fraud-risk-api.onrender.com/monitoring/operational
```
**Expected Response:**
- Returns total predictions logged, score decile distribution, tier percentages, and review backlog.

---

## 4. Free-Tier Characteristics & Cold-Start Handling

| Free-Tier Characteristic | Operational Impact | Mitigation & Design Standard |
|---|---|---|
| **Inactivity Spin-Down** | Service sleeps after 15 minutes of zero traffic. | Initial wake-up takes 30–50 seconds. The Next.js frontend handles this by displaying a clean loading pulse on cold start. |
| **Memory Limit (512 MB)** | Free tier instance memory is capped at 512 MB. | Model artifacts are loaded strictly once at startup (`app.state`), using ~80 MB RAM total. |
| **Supabase Free Pausing** | Supabase database pauses after 7 days without queries. | Resilient In-Memory fallback in `api/db.py` ensures the API never returns 500 even if the database is paused. |
| **Claim Standard (§10)** | Avoids misleading "production 99.99% SLA" phrasing. | Documented as: *"Built entirely on free tiers of hosted services, each with its own usage limits and inactivity behavior."* |
