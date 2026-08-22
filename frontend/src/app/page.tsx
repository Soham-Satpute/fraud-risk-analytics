"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { ScrollspyNav, NavSection } from "@/components/ScrollspyNav";
import { PowerBIEmbed } from "@/components/PowerBIEmbed";
import {
  ArrowRight,
  ExternalLink,
  FileText,
  ChevronRight,
  Lightbulb,
  Target,
  AlertTriangle,
  TrendingUp,
  Shield,
  DollarSign,
  Users,
  Zap,
  BarChart2,
  Search,
  ClipboardCheck,
} from "lucide-react";

const SECTIONS: NavSection[] = [
  { id: "overview", label: "The Story" },
  { id: "problem", label: "The Problem" },
  { id: "investigation", label: "What I Found" },
  { id: "model-results", label: "The Result" },
  { id: "business-decision", label: "The Decision" },
  { id: "dashboard", label: "Proof & Evidence" },
  { id: "robustness", label: "How Reliable?" },
  { id: "recommendation", label: "Final Answer" },
];

// ── Animated counter ──────────────────────────────────────────────────────────
function AnimatedNumber({
  target,
  prefix = "",
  suffix = "",
  decimals = 0,
}: {
  target: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}) {
  const [current, setCurrent] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          const duration = 1400;
          const start = performance.now();
          const animate = (now: number) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCurrent(eased * target);
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
          observer.disconnect();
        }
      },
      { threshold: 0.3 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target]);
  return (
    <span ref={ref}>
      {prefix}{current.toFixed(decimals)}{suffix}
    </span>
  );
}

// ── Plain-English Insight Block ───────────────────────────────────────────────
function InsightBlock({
  icon,
  label,
  headline,
  plain,
  detail,
  color = "var(--accent-blue)",
}: {
  icon: React.ReactNode;
  label: string;
  headline: string;
  plain: string;
  detail?: string;
  color?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="card"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        transition: "transform 180ms ease, box-shadow 180ms ease",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = "translateY(-3px)";
        (e.currentTarget as HTMLDivElement).style.boxShadow = "0 8px 24px rgba(0,0,0,0.35)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = "translateY(0)";
        (e.currentTarget as HTMLDivElement).style.boxShadow = "";
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <div
          style={{
            width: 36, height: 36,
            borderRadius: "var(--radius-md)",
            background: `${color}18`,
            border: `1px solid ${color}30`,
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0, color,
          }}
        >
          {icon}
        </div>
        <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {label}
        </span>
      </div>
      <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.35 }}>
        {headline}
      </h3>
      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
        {plain}
      </p>
      {detail && (
        <>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            style={{
              background: "none", border: "none", cursor: "pointer",
              fontSize: "0.75rem", fontWeight: 600, color,
              display: "flex", alignItems: "center", gap: "4px", padding: 0,
            }}
          >
            {open ? "Hide technical detail \u2191" : "Show how I handled it \u2192"}
          </button>
          {open && (
            <div
              className="fade-in"
              style={{
                background: "var(--bg-app)",
                border: `1px solid ${color}30`,
                borderRadius: "var(--radius-sm)",
                padding: "10px 12px",
                fontSize: "0.78rem",
                color: "var(--text-secondary)",
                lineHeight: 1.55,
              }}
            >
              {detail}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Tier Card ─────────────────────────────────────────────────────────────────
function TierCard({
  tier, label, score, description, color, bg, border, volume,
}: {
  tier: string; label: string; score: string; description: string;
  color: string; bg: string; border: string; volume: string;
}) {
  return (
    <div style={{ padding: "18px", background: bg, border: `1px solid ${border}`, borderRadius: "var(--radius-lg)", display: "flex", flexDirection: "column", gap: "8px" }}>
      <div style={{ fontSize: "0.7rem", fontWeight: 700, color, textTransform: "uppercase", letterSpacing: "0.06em" }}>{tier}</div>
      <div style={{ fontSize: "1.25rem", fontWeight: 800, color, fontFamily: "var(--font-mono)" }}>{label}</div>
      <div style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{score}</div>
      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55, marginTop: "2px" }}>{description}</p>
      <div style={{ marginTop: "4px", fontSize: "0.72rem", fontWeight: 700, color: "var(--text-muted)" }}>{volume}</div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function StoryPage() {
  return (
    <div className="story-layout">
      <aside className="scrollspy-sidebar">
        <ScrollspyNav sections={SECTIONS} />
      </aside>

      <article style={{ minWidth: 0 }}>

        {/* ── SECTION 1: THE STORY ── */}
        <section id="overview" className="story-section">
          <div style={{ marginBottom: "8px" }}>
            <span className="chip chip-blue" style={{ fontSize: "0.72rem" }}>
              Portfolio Project · End-to-End Analytics Case Study
            </span>
          </div>
          <h1 style={{ fontSize: "2.2rem", fontWeight: 800, lineHeight: 1.2, letterSpacing: "-0.025em", marginBottom: "16px", marginTop: "12px" }}>
            I built a system that spots financial fraud,{" "}
            <span style={{ color: "var(--accent-blue)" }}>
              and proved exactly how much money it saves.
            </span>
          </h1>

          <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderLeft: "4px solid var(--accent-blue)", borderRadius: "var(--radius-lg)", padding: "20px 24px", marginBottom: "32px", maxWidth: "740px" }}>
            <p style={{ fontSize: "1.05rem", color: "var(--text-primary)", lineHeight: 1.7 }}>
              Imagine you run an online store and 3.5% of your transactions are fraudulent, costing you thousands every day.
              You cannot review every purchase manually (too expensive), and you cannot just block suspicious ones blindly (you would annoy real customers).
            </p>
            <p style={{ fontSize: "1.05rem", color: "var(--text-primary)", lineHeight: 1.7, marginTop: "12px" }}>
              This project builds a machine learning system that reads through{" "}
              <strong>590,540 real financial transactions</strong>, learns what fraud looks like, and routes each new transaction to the right response, automatically.
              The result: an estimated{" "}
              <strong style={{ color: "var(--risk-low)" }}>$649,433 in savings</strong>{" "}
              while sending <strong>83.5% fewer</strong> transactions to expensive human reviewers.
            </p>
          </div>

          <div className="stat-grid" style={{ marginBottom: "32px" }}>
            <div className="card" style={{ background: "linear-gradient(135deg, var(--bg-card) 0%, rgba(76,142,237,0.08) 100%)", borderColor: "var(--accent-blue-border)" }}>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em", marginBottom: "6px" }}>Money Saved</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--risk-low)", fontFamily: "var(--font-mono)", lineHeight: 1 }}>
                <AnimatedNumber target={649433} prefix="$" />
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>Net financial value over doing nothing</div>
            </div>
            <div className="card">
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em", marginBottom: "6px" }}>Fraud Caught</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--accent-blue)", fontFamily: "var(--font-mono)", lineHeight: 1 }}>
                <AnimatedNumber target={65.95} suffix="%" decimals={1} />
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>Of all fraud transactions identified</div>
            </div>
            <div className="card">
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em", marginBottom: "6px" }}>Work Reduced</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--text-primary)", fontFamily: "var(--font-mono)", lineHeight: 1 }}>
                <AnimatedNumber target={83.5} suffix="%" decimals={1} />
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>Fewer cases sent to human reviewers</div>
            </div>
            <div className="card">
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em", marginBottom: "6px" }}>Model Quality</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--risk-low)", fontFamily: "var(--font-mono)", lineHeight: 1 }}>
                2x <span style={{ fontSize: "1rem" }}>better</span>
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>Than the standard baseline approach</div>
            </div>
          </div>

          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <Link href="/demo" className="btn btn-primary" style={{ padding: "11px 20px" }}>
              <Zap size={15} />
              <span>See it score live transactions</span>
              <ArrowRight size={14} />
            </Link>
            <a href="https://github.com/Soham-Satpute/fraud-risk-analytics/blob/main/case-study/fraud-risk-case-study.md" target="_blank" rel="noopener noreferrer" className="btn" style={{ padding: "11px 20px" }}>
              <FileText size={15} />
              <span>Full Case Study</span>
              <ExternalLink size={12} opacity={0.7} />
            </a>
          </div>
        </section>

        {/* ── SECTION 2: THE PROBLEM ── */}
        <section id="problem" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Why This Problem Is Hard</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "14px", lineHeight: 1.25 }}>
            Fraud is a needle-in-a-haystack problem at scale
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "28px" }}>
            Out of every 100 transactions, only{" "}
            <strong style={{ color: "var(--text-primary)" }}>3.5 are fraudulent</strong>.
            That sounds small, but across millions of transactions a day, that is an enormous amount of money being stolen.
            The challenge is not just finding fraud. It is doing so without crying wolf on legitimate customers and without breaking the bank on manual reviews.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "14px", marginBottom: "24px" }}>
            <div className="card" style={{ borderColor: "var(--risk-high-border)", background: "var(--risk-high-bg)" }}>
              <div style={{ fontSize: "1.5rem", marginBottom: "8px" }}>😰</div>
              <h3 style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--risk-high)", marginBottom: "6px" }}>Miss the fraud</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                Every undetected fraud costs <strong>~$150</strong> in chargebacks. Multiply by thousands and it becomes a business-critical loss.
              </p>
            </div>
            <div className="card" style={{ borderColor: "var(--risk-med-border)", background: "var(--risk-med-bg)" }}>
              <div style={{ fontSize: "1.5rem", marginBottom: "8px" }}>💸</div>
              <h3 style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--risk-med)", marginBottom: "6px" }}>Over-review everything</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                Manually reviewing every suspicious transaction costs <strong>$8 per case</strong>. Reviewing all test transactions would cost <strong>$944,864</strong>, which is more than the fraud itself.
              </p>
            </div>
            <div className="card" style={{ borderColor: "var(--accent-blue-border)", background: "var(--accent-blue-bg)" }}>
              <div style={{ fontSize: "1.5rem", marginBottom: "8px" }}>✅</div>
              <h3 style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--accent-blue)", marginBottom: "6px" }}>The sweet spot</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                Route only the <em>most suspicious</em> transactions to humans, automatically challenge medium-risk ones, and approve the clear majority instantly. This is what this system does.
              </p>
            </div>
          </div>

          <div className="card" style={{ background: "var(--bg-surface)", display: "flex", gap: "20px", alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 260px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                <BarChart2 size={16} color="var(--accent-blue)" />
                <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase" }}>The Dataset</span>
              </div>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                I worked with the{" "}
                <strong style={{ color: "var(--text-primary)" }}>IEEE-CIS Fraud Detection</strong>{" "}
                dataset: 590,540 real financial transactions spanning 26 weeks, with 434 features per transaction describing card details, amounts, device info, and behavioral patterns.
              </p>
            </div>
            <div style={{ flex: "1 1 260px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {[
                { label: "Total transactions", value: "590,540" },
                { label: "Fraud rate", value: "3.5%" },
                { label: "Data columns", value: "434" },
                { label: "Time window", value: "26 weeks" },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "10px 12px" }}>
                  <div style={{ fontSize: "1.1rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{value}</div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "2px" }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── SECTION 3: WHAT I FOUND ── */}
        <section id="investigation" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Before I Built Anything</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "12px", lineHeight: 1.25 }}>
            I spent weeks auditing the data before writing a single model
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "28px" }}>
            Most fraud models fail in production not because the algorithm is wrong, but because the data preparation was sloppy.
            I treated this phase as a detective investigation, looking for traps, biases, and hidden assumptions that would make results look good on paper but fail in the real world.
          </p>

          <div className="finding-grid">
            <InsightBlock
              icon={<Search size={16} />}
              label="Trap #1, Data Leakage"
              headline="The way you split your data can make results look 47% better than they actually are."
              plain="If you test your model on transactions from the same customers it trained on, it looks amazing, but it is cheating. I enforced a strict time-based split: train on earlier data, test on later data, exactly as it would work in the real world."
              detail="Entity overlap analysis revealed 74.7% card overlap under random split vs. 67.6% under temporal split. I validated the temporal split at TransactionDT <= 12,192,854 (80th percentile). I also ran a zero-overlap stress test on 10,952 completely unseen card identities."
              color="var(--risk-high)"
            />
            <InsightBlock
              icon={<Lightbulb size={16} />}
              label="Discovery #1, Free Features"
              headline="Two of the most predictive signals were already in the data, hiding in plain sight."
              plain="One column (D1) already measures how long since this card was last used. A major fraud signal. Another (C1) already counts how many times this card transacted recently. I spotted this and skipped rebuilding them, saving weeks of engineering work."
              detail="D1 has |r| = 1.000 with the engineered time-since-last-transaction feature. C1 has |r| = 1.000 with transaction velocity. Using duplicates would add noise, not signal."
              color="var(--risk-low)"
            />
            <InsightBlock
              icon={<AlertTriangle size={16} />}
              label="Trap #2, Misleading Explanations"
              headline="339 data columns were nearly identical copies of each other, which breaks fraud explanations."
              plain="When the system explains why a transaction looks suspicious, it needs to be accurate. If 162 pairs of near-identical data columns each claim partial credit, the explanation becomes meaningless. I detected and merged these duplicates so every explanation is clean and defensible."
              detail="162 V-feature pairs show |r| >= 0.98. I built a collinearity consolidation engine in reason_codes.py that groups them into unified driver clusters before producing analyst summaries."
              color="var(--risk-med)"
            />
            <InsightBlock
              icon={<TrendingUp size={16} />}
              label="Discovery #2, Fraud Grows Over Time"
              headline="Fraud rate increased from 3.40% to 3.61% across the 26-week window. Attackers escalate."
              plain="Fraudsters do not stand still. I measured this drift and designed the decision policy to be cost-sensitive, so the system can be re-tuned as conditions change without rebuilding the whole model."
              detail="PSI > 0.10 detected on D4, D6, D10, D14, D15. The 36-scenario sensitivity matrix was built specifically to test threshold stability under this kind of drift."
              color="var(--accent-blue)"
            />
            <InsightBlock
              icon={<Users size={16} />}
              label="Discovery #3, Missing Data Pattern"
              headline="76% of transactions have no device or browser information, but the absence itself is a signal."
              plain="When a fraudster uses an anonymous device, there is no identity record. I turned this missing data into a feature: the fact that identity data is missing becomes evidence that something might be off. That is a bit like noticing someone paid cash at a store that only accepts cards."
              detail="Only 23.8% of transactions successfully join to the identity table. Modeled missingness with indicator flags and frequency encodings, yielding 24 new engineered features (458 total)."
              color="var(--text-secondary)"
            />
            <InsightBlock
              icon={<Target size={16} />}
              label="Discovery #4, Night Owls"
              headline="Fraud spikes 1.36x during overnight hours (1am to 5am)."
              plain="Real customers mostly shop during the day. Fraudsters often run scripts overnight when fewer people are watching. I built time-of-day features that capture this, giving the model a sense of suspicion based on when the transaction happens."
              detail="Diurnal signal: 1.36x Risk Ratio for 1am-5am window. Self-transfer recipients: 3.30x RR with 9.29% fraud rate. Email domain mismatch: 3.55x RR. All computed with 95% Wilson Score Confidence Intervals."
              color="var(--accent-blue)"
            />
          </div>
        </section>

        {/* ── SECTION 4: THE RESULT ── */}
        <section id="model-results" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>How Well Does It Work?</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "12px", lineHeight: 1.25 }}>
            The model is almost twice as good as the standard approach
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "28px" }}>
            I compared my model (LightGBM, a state-of-the-art algorithm) against a standard baseline (Logistic Regression, the industry&apos;s typical first approach).
            The test was done on{" "}
            <strong style={{ color: "var(--text-primary)" }}>118,108 future transactions</strong>{" "}
            the model had never seen. Think of this as the final exam.
          </p>

          <div className="table-wrapper" style={{ marginBottom: "24px" }}>
            <table>
              <thead>
                <tr>
                  <th>What we measured</th>
                  <th style={{ color: "var(--accent-blue)" }}>My Model</th>
                  <th style={{ color: "var(--text-muted)" }}>Standard Approach</th>
                  <th style={{ color: "var(--risk-low)" }}>Improvement</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: "var(--accent-blue-bg)" }}>
                  <td>
                    <strong>Overall Ranking Quality</strong>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>How well does it rank fraud above legitimate? (Higher = better)</div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-blue)" }}>0.5441</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>0.2746</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>+98% better</td>
                </tr>
                <tr>
                  <td>
                    <strong>Fraud caught at strict budget</strong>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>If we allow 1% false alarms, how many real frauds do we catch?</div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>46.6%</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>15.1%</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>3x more fraud caught</td>
                </tr>
                <tr>
                  <td>
                    <strong>Fraud caught at moderate budget</strong>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>If we allow 5% false alarms, how many real frauds do we catch?</div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-primary)" }}>65.9%</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>41.8%</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>+24 percentage points</td>
                </tr>
                <tr>
                  <td>
                    <strong>General discrimination ability</strong>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>Can it tell fraud from legitimate at all cutoffs? (90%+ = excellent)</div>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>90.4%</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>80.9%</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>+12% lift</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="card" style={{ borderLeft: "3px solid var(--risk-low)", background: "rgba(63,185,80,0.05)", marginBottom: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <Shield size={16} color="var(--risk-low)" />
              <strong style={{ fontSize: "0.9rem" }}>Operational Impact</strong>
            </div>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
              The standard approach sends{" "}
              <strong style={{ color: "var(--text-primary)" }}>26,089 transactions</strong>{" "}
              to human reviewers. My model sends only{" "}
              <strong style={{ color: "var(--risk-low)" }}>4,297</strong>{" "}
              , while catching <em>more</em> fraud. That is the equivalent of going from needing a full review team to needing just a small squad.
            </p>
          </div>

          <div className="card" style={{ borderLeft: "3px solid var(--risk-med)", background: "rgba(210,153,34,0.05)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <AlertTriangle size={15} color="var(--risk-med)" />
              <strong style={{ fontSize: "0.82rem", color: "var(--risk-med)", textTransform: "uppercase" }}>Honest Limitation</strong>
            </div>
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              When tested on <strong>completely new cards never seen in training</strong> (the real worst-case scenario), performance drops by ~17.5%.
              This is expected; no model is perfect against truly novel attackers. The decision policy explicitly accounts for this by adding a second layer of automated verification for medium-risk cases.
            </p>
          </div>
        </section>

        {/* ── SECTION 5: THE DECISION ── */}
        <section id="business-decision" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Turning the Model into a Business Decision</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "12px", lineHeight: 1.25 }}>
            A good score alone does not save money. The right decision policy does
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "28px" }}>
            Here is where this project goes beyond most ML portfolios. I did not just build a model and show accuracy scores.
            I designed a{" "}
            <strong style={{ color: "var(--text-primary)" }}>3-tier routing system</strong>{" "}
            that decides what to <em>do</em> with each score, and proved mathematically which policy saves the most money.
          </p>

          <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "20px 24px", marginBottom: "28px", display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <Lightbulb size={20} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ fontSize: "0.9rem", display: "block", marginBottom: "6px" }}>Think of it like airport security lanes:</strong>
              <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
                A PreCheck traveler (low risk) walks straight through. A regular traveler (medium risk) goes through the standard scanner.
                A flagged person (high risk) gets the full investigation. The goal is maximum safety with minimum disruption to innocent passengers, and minimal cost to the airport.
              </p>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "14px", marginBottom: "28px" }}>
            <TierCard
              tier="Tier 1, Green Lane"
              label="Approve Instantly"
              score="Score < 0.01 (Very low risk)"
              description="The transaction looks clean. Approve it immediately with zero friction to the customer. Fast, free, and good for business."
              color="var(--risk-low)"
              bg="var(--risk-low-bg)"
              border="var(--risk-low-border)"
              volume="~78% of all transactions"
            />
            <TierCard
              tier="Tier 2, Yellow Lane"
              label="Extra Verification"
              score="Score 0.01 to 0.70 (Moderate risk)"
              description="Something looks slightly off. Send an automated OTP or 3D-Secure challenge ($0.50 cost). Most legitimate customers pass easily. Fraudsters usually do not."
              color="var(--risk-med)"
              bg="var(--risk-med-bg)"
              border="var(--risk-med-border)"
              volume="~18% of all transactions"
            />
            <TierCard
              tier="Tier 3, Red Lane"
              label="Human Review Queue"
              score="Score >= 0.70 (High risk)"
              description="The model is very confident this is suspicious. Route it to a fraud analyst for manual review ($8.00 cost). Over 62% of these are real fraud, an extremely high hit rate."
              color="var(--risk-high)"
              bg="var(--risk-high-bg)"
              border="var(--risk-high-border)"
              volume="Only 3.6% of all transactions"
            />
          </div>

          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", color: "var(--text-primary)" }}>
            I tested 3 policies and selected the mathematically optimal one:
          </h3>
          <div className="table-wrapper" style={{ marginBottom: "20px" }}>
            <table>
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Strategy</th>
                  <th>Cases reviewed</th>
                  <th>Total cost</th>
                  <th style={{ color: "var(--risk-low)" }}>Net savings</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <strong>Conservative</strong>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Minimal review team needed</div>
                  </td>
                  <td style={{ fontSize: "0.82rem" }}>Only review top 1% most suspicious</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>1,181</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>$232,042</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 700 }}>$525,487</td>
                </tr>
                <tr style={{ background: "var(--accent-blue-bg)" }}>
                  <td>
                    <strong style={{ color: "var(--accent-blue)" }}>Balanced (RECOMMENDED)</strong>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Best return on review capacity</div>
                  </td>
                  <td style={{ fontSize: "0.82rem" }}>Review top 5% most suspicious</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>4,297</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>$108,096</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 800, fontSize: "1rem" }}>$649,433</td>
                </tr>
                <tr>
                  <td>
                    <strong>Standard baseline</strong>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Typical industry starting point</div>
                  </td>
                  <td style={{ fontSize: "0.82rem" }}>Default 50% threshold, uncapped</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>26,089</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>$422,041</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>$335,488</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="card">
            <h4 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: "8px" }}>Why the Balanced policy wins</h4>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
              Reviewing all 118,108 transactions would cost <strong>$944,864</strong> in analyst time to recover <strong>$609,600</strong> in fraud, a net loss.
              Reviewing nothing costs $609,600 in chargebacks. Policy B sits exactly at the mathematical cost minimum: spend $108K, save $757K,
              net <strong style={{ color: "var(--risk-low)" }}>$649K</strong>.
            </p>
          </div>
        </section>

        {/* ── SECTION 6: DASHBOARD ── */}
        <section id="dashboard" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Proof and Observability</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "12px", lineHeight: 1.25 }}>
            I built a live dashboard so the business can always see what is happening
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "20px" }}>
            A model that runs invisibly is a black box that nobody trusts. I connected the system to Power BI so executives, fraud analysts, and ops teams each get the view they need.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", marginBottom: "20px" }}>
            {[
              { num: "1", label: "Financial ROI chart", desc: "Shows fraud losses prevented vs. review costs. At a glance: is the system making money?", color: "var(--risk-low)" },
              { num: "2", label: "Model health gauges", desc: "Precision (54%) and Recall (79%) meters with green/amber/red status. No data science degree required.", color: "var(--accent-blue)" },
              { num: "3", label: "High-risk review queue", desc: "Live list of flagged transactions ranked by fraud probability, with a plain-English reason for each flag.", color: "var(--risk-high)" },
            ].map(({ num, label, desc, color }) => (
              <div key={num} className="card" style={{ padding: "14px 16px", borderTop: `3px solid ${color}` }}>
                <div style={{ fontSize: "0.7rem", fontWeight: 700, color, textTransform: "uppercase", marginBottom: "6px" }}>Panel {num}</div>
                <div style={{ fontSize: "0.875rem", fontWeight: 700, marginBottom: "6px" }}>{label}</div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>{desc}</p>
              </div>
            ))}
          </div>

          <PowerBIEmbed imageSrc="/dashboard-powerbi.png" pdfUrl="/Fraud-risk-analytics.pdf" />
        </section>

        {/* ── SECTION 7: ROBUSTNESS ── */}
        <section id="robustness" className="story-section">
          <span className="chip chip-grey" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Is This Actually Reliable?</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "12px", lineHeight: 1.25 }}>
            I stress-tested the recommendation across 36 different what-if scenarios
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.7, maxWidth: "700px", marginBottom: "28px" }}>
            Any business decision can look good under the best assumptions. The real test is whether it holds up when things go wrong.
            What if fraud losses are smaller than we think? What if review costs double? What if we have fewer analysts available?
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "14px", marginBottom: "24px" }}>
            {[
              { scenario: "Optimistic", desc: "High fraud losses ($200), cheap reviews ($5), generous capacity", savings: "$982,540", verdict: "Policy B still wins", color: "var(--risk-low)", highlighted: false },
              { scenario: "Base Case", desc: "$150 fraud loss, $8 review cost, 5% capacity cap", savings: "$649,433", verdict: "Policy B recommended", color: "var(--accent-blue)", highlighted: true },
              { scenario: "Pessimistic", desc: "Low fraud losses ($100), expensive reviews ($12), tight capacity", savings: "$332,190", verdict: "More conservative tier applies", color: "var(--risk-med)", highlighted: false },
            ].map(({ scenario, desc, savings, verdict, color, highlighted }) => (
              <div key={scenario} className="card" style={{ background: highlighted ? "var(--accent-blue-bg)" : undefined, borderColor: highlighted ? "var(--accent-blue-border)" : undefined }}>
                <div style={{ fontSize: "0.72rem", fontWeight: 700, color, textTransform: "uppercase", marginBottom: "6px" }}>{scenario}</div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: "10px" }}>{desc}</p>
                <div style={{ fontSize: "1.3rem", fontWeight: 800, fontFamily: "var(--font-mono)", color }}>{savings}</div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>{verdict}</div>
              </div>
            ))}
          </div>

          <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <ClipboardCheck size={16} color="var(--risk-low)" />
              <strong style={{ fontSize: "0.9rem" }}>Robustness Verdict</strong>
            </div>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
              Across all 36 scenarios tested, varying fraud loss assumptions, review costs, and staff capacity, the recommended decision threshold remained stable.
              Under <strong>no realistic conditions</strong> does it make sense to revert to the baseline approach or flood analysts with uncapped review queues.
            </p>
          </div>
        </section>

        {/* ── SECTION 8: FINAL ANSWER ── */}
        <section id="recommendation" className="story-section">
          <span className="chip chip-blue" style={{ fontSize: "0.7rem", marginBottom: "12px" }}>Final Answer</span>
          <h2 style={{ fontSize: "1.55rem", fontWeight: 800, marginBottom: "20px", lineHeight: 1.25 }}>
            The recommendation, in plain English
          </h2>

          <div className="verdict-box" style={{ marginBottom: "28px" }}>
            <p style={{ fontSize: "1.05rem", color: "var(--text-primary)", lineHeight: 1.75, marginBottom: "20px" }}>
              <strong>Deploy the Balanced Policy</strong>: route all financial transactions through the 3-tier system.
              Automatically approve the obvious ones, automatically challenge the borderline ones, and send only the most suspicious 3.6% to a human reviewer.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "14px", marginBottom: "20px" }}>
              {[
                { label: "Fraud caught", value: "65.9%", sub: "of all fraud transactions", color: "var(--risk-low)" },
                { label: "Cases to review", value: "4,297", sub: "down from 26,089", color: "var(--accent-blue)" },
                { label: "Net value", value: "$649K", sub: "vs $335K baseline", color: "var(--risk-low)" },
              ].map(({ label, value, sub, color }) => (
                <div key={label} style={{ background: "var(--bg-card)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)", padding: "14px 16px" }}>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, marginBottom: "4px" }}>{label}</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 800, fontFamily: "var(--font-mono)", color }}>{value}</div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>{sub}</div>
                </div>
              ))}
            </div>

            <div style={{ borderTop: "1px solid var(--border)", paddingTop: "18px", display: "flex", flexDirection: "column", gap: "10px" }}>
              <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em" }}>
                Why this recommendation is defensible:
              </p>
              {[
                { icon: <Users size={14} />, text: "Analysts review only 3.6% of volume, giving a manageable queue that does not cause burnout, with 62% of those cases being real fraud (a high hit rate)." },
                { icon: <Shield size={14} />, text: "The automated verification layer (Tier 2) protects against new fraudsters who have never been seen before, filling the gap the model alone cannot cover." },
                { icon: <DollarSign size={14} />, text: "The recommendation is backed by a 36-scenario stress test, not a single optimistic assumption. It holds across pessimistic, realistic, and optimistic conditions." },
              ].map(({ icon, text }, i) => (
                <div key={i} style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
                  <div style={{ width: 24, height: 24, borderRadius: "var(--radius-sm)", background: "var(--accent-blue-bg)", border: "1px solid var(--accent-blue-border)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "var(--accent-blue)", marginTop: "1px" }}>
                    {icon}
                  </div>
                  <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>{text}</p>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", gap: "14px", flexWrap: "wrap" }}>
            <Link href="/demo" className="btn btn-primary" style={{ padding: "13px 22px", fontSize: "0.95rem" }}>
              <Zap size={16} />
              <span>Watch it score 1,500 real transactions</span>
              <ChevronRight size={16} />
            </Link>
            <a href="https://github.com/Soham-Satpute/fraud-risk-analytics/blob/main/case-study/fraud-risk-case-study.md" target="_blank" rel="noopener noreferrer" className="btn" style={{ padding: "13px 22px", fontSize: "0.95rem" }}>
              <FileText size={16} />
              <span>Read Full Case Study</span>
              <ExternalLink size={14} opacity={0.7} />
            </a>
          </div>
        </section>

      </article>
    </div>
  );
}
