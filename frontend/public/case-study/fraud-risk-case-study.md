# Fraud Risk Analytics & Detection System
## A Case Study in Evidence-Driven Machine Learning

> **Audience:** Hiring Managers, Recruiters, Data Science Leaders
> **Project:** End-to-end fraud risk analytics system built on IEEE-CIS e-commerce data
> **Infrastructure:** 100% free-tier (Supabase, Render, Vercel, local LLM via Ollama)
> **Repository:** `Soham-Satpute/fraud-risk-analytics`

---

## The One-Line Summary

> I built a fraud risk scoring system that — on real held-out data — saves **$649,433** in net financial losses, cuts the fraud analyst review queue by **83.5%**, and explains every decision in plain English. The entire system runs for free.

---

## 01 — The Problem: Why Fraud Detection Is Harder Than It Looks

Imagine I run an e-commerce platform processing thousands of transactions per day. About **3.5% of them are fraud** — but I don't know which ones until it's too late.

Here's the operational dilemma I face:

**If I do nothing:** Every fraudulent transaction goes through. I absorb the loss directly — chargebacks, merchandise write-offs, penalties. On 118,000 test transactions, that's **$812,800 in losses**.

**If I flag everything manually:** My fraud analysts are buried. Reviewing 22% of all transactions — which is what happens with a naive model — costs more than the fraud I catch. Analysts burn out, miss real fraud in the noise, and the system collapses.

**The real question:** *Can I build a system that flags the right transactions, in the right volume, at the right threshold — and actually justify that choice with numbers?*

That's what this project set out to answer. Not just "train a model and report accuracy," but: **"What decision should the business actually make, and why?"**

---

## 02 — The Dataset: What I Was Working With

I used the **IEEE-CIS Fraud Detection dataset** — a real, publicly available dataset from a major e-commerce fraud detection challenge. No synthetic data, no toy examples.

| What I Had | Details |
|---|---|
| Total transactions | 590,540 rows |
| Features per transaction | 434 columns |
| Fraud rate | 3.499% (about 1 in 29 transactions) |
| Time span | 182 days (26 weeks) of transaction history |

The dataset is messy in ways that mirror real-world fraud data:
- Most transactions are **not labeled with customer identity** — only 23.8% have identity records attached
- Hundreds of features are **partially or heavily missing** (some up to 85% empty)
- The timestamps are **relative seconds from an undisclosed starting point** — I can't reconstruct an actual calendar date

These aren't bugs. They're properties of real financial data, and they forced genuine engineering decisions at every step.

---

## 03 — The Investigation: What I found Before Building Anything

> **The most important thing I did was spend a full week investigating the data before writing a single line of modeling code.**

Most ML projects skip straight to training a model. This project started with a forensic audit of the dataset. Here's what I found — and how each finding changed what I built.

---

### Finding 1: The Timestamps Are Relative, Not Real

The timestamp column (`TransactionDT`) doesn't represent a real calendar date. It represents **seconds elapsed since some undisclosed internal origin point**.

**Why this matters:** I couldn't use "December transactions" or "holiday season" as a feature — the calendar origin is simply unknown. Any code that assumed the timestamps were real clock times would be silently wrong.

**What I did instead:** I extracted the *cycle structure* from relative intervals — "what hour of the 24-hour daily cycle is this transaction in?" and "what day of the 7-day weekly cycle?" — without inventing a calendar anchor that doesn't exist.

---

### Finding 2: How You Split the Data Changes the Entire Answer

This was the most consequential discovery of the investigation. There are three common ways to split historical data into training and test sets:

| Split Strategy | What It Does | Entity Overlap | Problem |
|---|---|---|---|
| **Random split** | Shuffles all rows, takes 80% for training | 74.7% | Artificially easy — the model sees most customers during training and then "recognizes" them in the test set |
| **Temporal split** | Trains on early months, tests on later months | 67.6% | Realistic — mirrors how a deployed model actually works: trained on the past, evaluated on the future |
| **Grouped entity split** | Ensures zero customer overlap between train and test | 0% | Too conservative — breaks the time ordering entirely |

I measured what the random split would actually do to my results: it would inflate the precision-recall metric by **+24.8%** — making the model look dramatically better than it actually is.

**The decision I made:** Enforce a strict **temporal split** — train on the first 80% of the time window, test on the remaining 20%. This is how a real deployed model works, so this is how I measured it.

> **The key principle:** I didn't pick the split strategy that made my numbers look best. I picked the one that most honestly reflects the deployment reality.

---

### Finding 3: Some Features I Planned to Build Already Existed

The dataset contains 339 "V-features" (engineered by the payment processor, Vesta), 15 "D-features" (time-delta columns), and 14 "C-features" (transaction count columns).

I planned to engineer two features from scratch:
1. **Transaction velocity** — how many recent transactions has this card made?
2. **Time since last transaction** — how long since this card was last used?

After correlation analysis, I discovered that these features **already existed in the raw data** with perfect correlation (r = 1.000):
- `C1` **is** the transaction velocity feature
- `D1` **is** the time-since-last-transaction feature

**The decision I made:** Don't rebuild them. Use them directly. Instead, I focused engineering effort on features that *don't* already exist in the raw data — like amount z-scores and email domain consistency.

> **Why this matters:** It prevented data leakage (where the engineered feature and the raw feature would be treated as independent by the model, when they're actually identical), and it kept the feature set honest.

---

### Finding 4: Hundreds of V-Features Are Near-Identical

Among the 339 V-features, I found **162 pairs** where the correlation was 0.98 or higher — essentially the same information stored in two different column names.

**The decision I made:** Keep them all in the model (tree-based models handle redundancy fine), but **document the collinear groups** so that individual prediction explanations consolidate near-identical reasons rather than showing the analyst four entries that all say the same thing.

---

## 04 — The Engineering: Building a Leakage-Free System

With the investigation complete, I had a clear picture of what to build. Here's what I engineered and why each decision was made.

---

### Decision: Which Features to Build

I built **24 new features** — each chosen because it captured signal that wasn't already in the raw columns:

| Feature Category | What It Measures | Why It Matters |
|---|---|---|
| **Amount z-score** (by card, by card+address) | How unusual is this transaction amount *for this specific card*? A $3,000 transaction from a card that usually spends $30 is suspicious. A $3,000 transaction from a card that regularly spends $3,000 is not. | Relative deviation is more informative than absolute amount |
| **Log-transformed amount** | Compresses the huge range of transaction amounts into a more manageable scale | Transaction amounts span from <$1 to >$10,000 — log transform prevents the model from being dominated by extreme values |
| **Frequency encoding** (card, address, email domain) | How commonly does this card / address / email appear in the training data? Rare combinations are suspicious. | Low-frequency combinations are a fraud signal |
| **Cyclical time features** (hour-of-day, day-of-week) | What time cycle is this transaction in? | Fraud peaks at night; encoding this as a cycle (not a raw number) prevents the model from thinking "hour 23" and "hour 0" are far apart |
| **Email domain consistency** | Does the purchaser's email domain match the recipient's? | Self-transfers (same domain payer and recipient) have a **3.3x higher fraud rate** |

**Critical guardrail:** All frequency statistics and group averages were calculated *only from the training data* and then applied to the test set. This prevents any form of lookahead — the model never gets to "peek" at test transactions when learning what's normal.

---

### Decision: How to Handle Class Imbalance

Only 3.5% of transactions are fraud. That means if the model does nothing and just predicts "not fraud" every time, it would be right 96.5% of the time. That's useless.

I tested three approaches:

1. **Class weighting** — Tell the model that fraud cases are 27x more important than legitimate ones. Fast, no data distortion.
2. **SMOTE (synthetic minority oversampling)** — Generate fake fraud transactions to balance the training set. Risks distorting the true distribution.
3. **No adjustment (ablation test)** — Train without any imbalance handling to measure the baseline cost.

**The decision I made:** Use **class weighting** (`scale_pos_weight = 27.46`). The ablation test confirmed the weighted model outperformed the unweighted one on the precision-recall metric that matters for fraud (PR-AUC), and SMOTE was rejected because it creates synthetic fraud patterns that may not reflect real adversarial behavior.

---

## 05 — The Model: What I built and How Well It Works

I trained two models and compared them rigorously on the held-out test set (118,108 transactions from the last 20% of the time window — data the model had never seen during training).

### The Two Models

**Model 1 — Logistic Regression (Baseline):** A simple, interpretable model. Its job is to set a floor — any more complex model I build needs to be clearly better than this to justify the added complexity.

**Model 2 — LightGBM (Champion):** A gradient boosted tree model. These are the workhorses of industrial fraud detection — fast, accurate, and well-suited to tabular data with complex non-linear patterns.

### How I measured Performance

I didn't use accuracy (meaningless for 3.5% fraud) or even ROC-AUC alone. The primary metric was **PR-AUC** (Precision-Recall Area Under Curve) — which directly measures how well the model finds fraud without overwhelming analysts with false alarms.

I also ran **1,000 bootstrap resamples** to compute honest confidence intervals. A point estimate (e.g. "PR-AUC = 0.54") tells me one number. A confidence interval tells me the range: "I'm 95% confident the true score is between 0.53 and 0.56." This is the difference between a result and a defensible result.

### Results

| Metric | Champion LightGBM | Baseline Logistic Regression | What This Means |
|---|---|---|---|
| **PR-AUC** (primary) | **0.5441** `[0.5282—0.5607]` | 0.2746 `[0.2605—0.2891]` | LightGBM is **98% better** at ranking fraud without drowning analysts |
| **ROC-AUC** | **0.9035** `[0.8982—0.9087]` | 0.8092 `[0.8021—0.8164]` | Strong discrimination between fraud and legitimate transactions |
| **Fraud caught at 1% false alarm rate** | **46.6%** `[44.9%—48.2%]` | 15.1% | LightGBM catches **3x more fraud** while flagging the same number of legitimate transactions |
| **Fraud caught at 5% false alarm rate** | **65.9%** `[64.5%—67.5%]` | 41.8% | Even at a relaxed threshold, LightGBM catches 24 percentage points more fraud |

> **Plain English:** If I gave both models a list of transactions and told them "flag only the top 1% most suspicious" — LightGBM would catch 47 out of 100 frauds. The baseline would catch 15.

### The Generalization Test

To stress-test the model, I ran a separate evaluation on transactions from **completely new card entities** that the model had never seen during training (0% overlap). Performance degraded — PR-AUC dropped from 0.54 to 0.45 — but remained 63% better than the baseline. This "cold-start degradation" is expected and documented — and the system handles it by routing unfamiliar entities through the step-up authentication tier rather than auto-approving them.

---

## 06 — Explaining Decisions: How I Made the Model Understandable

A fraud score by itself is not useful for an analyst. "This transaction has an 87% fraud probability" tells the analyst nothing about *why* — and without a reason, they can't make a good decision or escalate appropriately.

I used **SHAP (SHapley Additive Explanations)** — a method grounded in game theory — to decompose the model's score into the contribution of each individual feature. For each transaction, I surface the top risk factors and top mitigating factors.

### The Consolidation Problem

The dataset's 339 V-features contained 162 near-duplicate pairs. Without consolidation, an analyst briefing might show:

> "Risk factor: V95 (SHAP: +0.31), V101 (SHAP: +0.30), V279 (SHAP: +0.29), V293 (SHAP: +0.28)"

That's four reasons that all say the same thing. I consolidated collinear groups into unified, named clusters:

> "Risk factor: **Payment Activity Volume** (a cluster of Vesta-encoded payment velocity signals) — significantly elevated, consistent with high-frequency card activity pattern"

Much more useful.

### The AI Explanation Layer

I added an optional layer that uses a **local LLM (running on Ollama — no cloud API, no cost)** to translate the SHAP reason codes into a short analyst briefing written in plain English.

**Critical safeguard:** Every generated narrative is run through an **automated grounding validator** before it reaches the analyst. The validator checks:

- Every number in the narrative matches the actual SHAP evidence (no hallucinated figures)
- Every claim about risk direction is consistent with the SHAP sign (no "this reduces risk" when SHAP says it increases risk)
- No speculation beyond what the evidence supports

**Result:** 93.5% of AI-generated narratives passed the validator on the first try. The remaining 6.5% were automatically replaced with a verified deterministic template. **Final grounding rate: 100%.**

> **Why this matters:** An AI that makes up fraud reasons is dangerous. The grounding validator is what turns "AI-generated text" into "evidence-backed analyst briefing."

---

## 07 — The Business Decision: Translating Model Results Into Action

This is the section most ML projects skip. They report PR-AUC and call it done. I didn't.

### The Core Question

The model produces a fraud probability for every transaction — a number between 0 and 1. But **what do I do with that number?**

- If p >= 0.50, flag it? That sends 22% of all transactions to analysts. Impossible to process.
- If p >= 0.90, flag it? Very precise, but I miss 77% of fraud.

The right threshold depends on:
1. **How much does each undetected fraud actually cost?**
2. **How much does each manual review cost?**
3. **How many reviews can the fraud team realistically handle per day?**

I modeled all of this explicitly.

### The 3-Tier Architecture

Rather than a single binary flag, I designed a **3-tier routing system**:

```
Every incoming transaction
           |
           v
   Model scores probability p
           |
    +------+------------------+
    |                         |
  p < 0.01              0.01 <= p < 0.70              p >= 0.70
    |                         |                         |
    v                         v                         v
TIER 1:                  TIER 2:                   TIER 3:
Auto-Approve             Step-Up Auth              Manual Review
(no friction)            (OTP / 3DS)               (analyst queue)
9.34% of volume          87.02% of volume          3.64% of volume
0.09% fraud rate         Stops 80% of             4,297 cases over
                         medium-risk fraud         26 weeks
                         automatically             51% are real fraud
```

**Tier 2 is the key innovation.** Instead of sending every suspicious transaction to a human, the system challenges it automatically with a second authentication step (like a one-time password or 3D-Secure verification). This stops ~80% of medium-risk fraud with no human involvement, at a cost of $0.50 per check.

### Comparing Policies

I evaluated 100 different threshold combinations and compared them against clear baselines:

| Policy | Fraud Caught | Analyst Reviews | Net Savings | Verdict |
|---|---|---|---|---|
| **Accept everything** (no model) | 0% | 0 | $0 | Starting point — unacceptable |
| **Flag amounts > $500** (naive rule) | 5.8% | 4,816 | $8,272 | Almost useless — misses 94% of fraud |
| **Logistic Regression at p >= 0.50** | 66.9% | 26,089 | $335,488 | Good recall, but queue overflow — 22% of all transactions |
| **Policy A (Conservative, tau = 0.96)** | 23.1% | 1,044 | $623,631 | High precision (90%), tiny queue — for capacity-constrained teams |
| **Policy B (Balanced, tau = 0.70)** <- Recommended | 53.9% | 4,297 | **$649,433** | Best financial outcome, manageable queue |

### Why Policy B Wins

Policy B's threshold (p >= 0.70 for manual review) was not chosen arbitrarily. It's the result of sweeping all 100 candidate thresholds and finding the one that minimizes total financial cost:

- **Fraud losses avoided:** The model's tier-2 and tier-3 routing prevents the majority of fraud
- **Review cost:** 4,297 reviews x $8/review = $34,376
- **Step-up authentication cost:** 102,782 checks x $0.50/check = $51,391
- **Remaining undetected fraud loss:** $78,240
- **Total operating cost:** $163,367 (vs. $812,800 with no model)
- **Net savings:** **$649,433**

Crucially, expanding the review queue beyond 3.64% doesn't help — the marginal review costs more than the marginal fraud it catches. The system naturally finds its own cost-efficient limit.

---

## 08 — Sensitivity Analysis: Does the Recommendation Hold Up?

A recommendation that only holds in one specific scenario isn't reliable. I tested my recommendation across **36 different scenarios**, varying:

- Fraud loss: $160, $200, or $240 per undetected fraud
- Review cost: $5, $8, or $12 per analyst investigation
- Team capacity: 1%, 3%, 5%, or 10% review rate limit

**Findings across all 36 scenarios:**

**Adopt Policy B — the Balanced 3-Tier Routing Architecture.**

Here's what that means operationally:

**For 9.3% of transactions (lowest risk):** Approve instantly. No friction for the customer. The model is confident these are legitimate.

**For 87% of transactions (medium risk):** Trigger an automated second authentication step — a one-time password, biometric check, or 3D-Secure challenge. The customer experiences a brief extra step; most legitimate customers complete it. Most fraudsters can't.

**For 3.6% of transactions (highest risk, p >= 0.70):** Route to a fraud analyst with a full briefing — SHAP reason codes, the AI-generated explanation, and the full transaction context. The analyst makes the final call.

**Three things to never do:**
1. **Never automatically decline transactions.** Decline decisions should only follow human review or failed step-up authentication. Automatic declines create false positives, customer churn, and legal exposure.
2. **Never use a 0.50 default threshold.** This sends 22% of transactions to analysts — an operationally impossible queue.
3. **Never assume this model is correct about a specific transaction.** The model gives probabilities, not certainties. Tier-3 routing exists precisely because the model isn't certain.

**When to revisit this recommendation:**
- If your team can only handle less than 1% of transactions for manual review -> Switch to Policy A (threshold = 0.96)
- If average fraud losses drop below $100/transaction (micro-fraud shift) -> Manual review becomes cost-inefficient; expand step-up authentication
- If step-up authentication costs rise above $1.50/check -> Raise the lower threshold from 0.01 to 0.05

---

## 10 — Decision Log: Every Key Choice, and Why

This section traces every major decision made in the project, for readers who want to see the reasoning behind the numbers.

| Decision | What I Considered | What I Chose | Why |
|---|---|---|---|
| **Validation split strategy** | Random split (74.7% entity overlap), temporal split (67.6%), grouped split (0%) | **Temporal split** | Random split inflates metrics by +24.8%; grouped split breaks time ordering. Temporal split mirrors real deployment. |
| **When to split** (train/test boundary) | Multiple percentiles tested | **80th percentile** of TransactionDT (<= 12,192,854) | Gives sufficient training history (472,432 rows) while keeping a meaningful test window (118,108 rows). |
| **Primary evaluation metric** | Accuracy, ROC-AUC, PR-AUC, F1 | **PR-AUC** | With 3.5% fraud, accuracy is misleading. PR-AUC directly measures the model's ability to find fraud without overwhelming analysts. ROC-AUC reported as secondary. |
| **Class imbalance handling** | SMOTE, class weighting, no adjustment | **Class weighting** (scale_pos_weight = 27.46) | Ablation confirmed it outperforms unweighted. SMOTE risks distorting the true fraud distribution with synthetic patterns. |
| **Model type** | Logistic Regression, XGBoost, LightGBM | **LightGBM** (champion) + Logistic Regression (baseline) | LightGBM achieved +98.1% PR-AUC lift over baseline. Baseline kept for honest comparison and interpretability reference. |
| **Velocity feature engineering** | Build rolling velocity counts from scratch | **Use raw C1 directly** | C1 is identical to the planned feature (r = 1.000). Building from scratch adds no value, creates confusion. |
| **Time-since-last feature** | Build from TransactionDT calculation | **Use raw D1 directly** | D1 is identical to the planned feature (r = 1.000). |
| **Feature fit boundary** | Fit on all data, or fit only on training data | **Fit frequency mappings and statistics only on training data** | Any statistics computed on test data create temporal lookahead leakage — the model would "know" future transaction volumes. |
| **V-feature collinear pairs** | Drop one of each pair, drop both, keep both | **Keep all, consolidate in SHAP display** | LightGBM handles redundancy. Dropping removes signal. But displaying all 162 duplicates in analyst briefings would be confusing — consolidate at the explanation layer. |
| **Operating threshold** | Use default 0.50 cutoff, pick highest accuracy, optimize for cost | **12-step cost matrix optimization across 100 candidate thresholds** | No single cutoff is universally correct. The right threshold depends on review capacity and cost assumptions. I modeled the full landscape. |
| **Routing architecture** | Binary flag (fraud / not fraud), single threshold, 3-tier | **3-tier routing** (auto-approve / step-up / manual review) | Step-up authentication stops the majority of medium-risk fraud automatically, without human cost. This is the key to making the queue manageable. |
| **LLM integration** | Live per-request cloud API, offline pre-generation, no LLM | **Offline batch generation via local Ollama** + **mandatory grounding validator** | No cloud API cost. No live dependency. Grounding validator makes the AI layer defensible rather than decorative. |
| **Postgres scope** | Store everything in Postgres, store nothing, store serving-layer only | **Serving/logging layer only** (predictions, model runs, demo replay) | Raw data (590K x 434 columns) would blow past Supabase's 500MB free tier. Model training needs fast columnar access — not a row-oriented database over a network. |

---

## 11 — Honest Limitations

No system is perfect. Here's what this one cannot do, and why:

**1. It degrades on customers it has never seen before.**
When a brand-new card entity appears — someone who has never transacted in the training period — the model's PR-AUC drops from 0.54 to 0.45. This "cold-start" degradation is expected and real. The system mitigates it by routing unfamiliar entities to step-up authentication rather than auto-approving them.

**2. The fraud cost assumptions are estimates, not ground truth.**
I modeled fraud losses at $200/transaction and review costs at $8/review. These are stated assumptions, not verified figures from a real institution's P&L. The sensitivity analysis shows the recommendation holds across a range of $160-$240 fraud loss and $5-$12 review cost — but an actual business would need to calibrate these to their own data.

**3. I can't reconstruct absolute calendar time.**
`TransactionDT` is a relative counter, not a real timestamp. Daily and weekly cycles can be modeled, but not "Christmas spike" or "Black Friday behavior." A production system would have real timestamps.

**4. Identity data is sparse.**
Only 23.8% of transactions have identity records attached. For the other 76.2%, the model must rely on card and amount signals alone.

**5. This is a portfolio-scale deployed demo, not a production system.**
The hosting runs on free tiers (Render, Supabase, Vercel) with cold-start delays and storage limits. A real fintech deployment would use dedicated infrastructure, real-time feature stores, and a proper MLOps monitoring stack.

---

## 12 — What's Actually Running

The complete system is deployed and accessible:

| Component | Technology | What It Does |
|---|---|---|
| **Fraud scoring API** | FastAPI on Render (free tier) | Accepts transaction data, returns risk probability, risk tier, and SHAP reason codes. Median response time: 314ms. |
| **Interactive demo** | Next.js on Vercel (free tier) | Replays 1,500 real held-out transactions through the live model. Shows risk scores, reason codes, and analyst narratives. |
| **Methodology page** | Next.js on Vercel | Documents the investigation findings, model benchmarks, and 36-scenario sensitivity matrix. |
| **Audit database** | PostgreSQL on Supabase (free tier) | Stores predictions, model run metadata, and the demo replay slice. Never stores raw training data. |
| **Model files** | Local/GitHub | Serialized LightGBM champion model and fitted feature pipeline. Reproducible from the repository. |
| **72 automated tests** | pytest | Covers data quality, feature engineering, model evaluation, grounding validation, business decision logic, and API endpoints. All passing. |

**Infrastructure cost: $0/month.** Built entirely on perpetual free tiers with no expiring trial credits.

---

*Built end-to-end by Soham Satpute. All numbers are real — computed on the actual IEEE-CIS dataset with zero placeholders. Source: `Soham-Satpute/fraud-risk-analytics`.*
