"use client";

import { useEffect, useState } from "react";
import { fetchBusinessDecisionData, BusinessDecisionData } from "@/lib/api";
import {
  BarChart3,
  ShieldCheck,
  TrendingUp,
  Cpu,
  Layers,
  CheckCircle2,
  DollarSign,
  AlertTriangle,
  FileCheck,
} from "lucide-react";

export default function MethodologyPage() {
  const [data, setData] = useState<BusinessDecisionData | null>(null);
  const [selectedTab, setSelectedTab] = useState<"policies" | "sensitivity" | "integrity">("policies");

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetchBusinessDecisionData();
        setData(res);
      } catch (err) {
        console.error("Failed to load business decision data:", err);
      }
    }
    loadData();
  }, []);

  return (
    <div className="container" style={{ paddingBottom: "60px" }}>
      {/* Header */}
      <div style={{ marginTop: "32px", marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
          <h1 style={{ fontSize: "1.8rem", fontWeight: 800, letterSpacing: "-0.03em" }}>
            Methodology & Analytics Summary
          </h1>
          <span className="badge badge-cyan">Statistical Rigor</span>
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
          Empirical validation findings, 1,000-resample bootstrap confidence intervals, 3-tier cost-sensitive threshold decisions, and multi-variable sensitivity matrices.
        </p>
      </div>

      {/* Navigation Tabs */}
      <div style={{
        display: "flex",
        gap: "10px",
        marginBottom: "24px",
        borderBottom: "1px solid var(--border-subtle)",
        paddingBottom: "12px",
      }}>
        <button
          onClick={() => setSelectedTab("policies")}
          style={{
            padding: "8px 18px",
            borderRadius: "var(--radius-md)",
            fontSize: "0.88rem",
            fontWeight: 600,
            cursor: "pointer",
            background: selectedTab === "policies" ? "var(--accent-cyan-bg)" : "transparent",
            color: selectedTab === "policies" ? "var(--accent-cyan)" : "var(--text-secondary)",
            border: selectedTab === "policies" ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
          }}
        >
          Candidate Policies & Baselines
        </button>

        <button
          onClick={() => setSelectedTab("sensitivity")}
          style={{
            padding: "8px 18px",
            borderRadius: "var(--radius-md)",
            fontSize: "0.88rem",
            fontWeight: 600,
            cursor: "pointer",
            background: selectedTab === "sensitivity" ? "var(--accent-cyan-bg)" : "transparent",
            color: selectedTab === "sensitivity" ? "var(--accent-cyan)" : "var(--text-secondary)",
            border: selectedTab === "sensitivity" ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
          }}
        >
          36-Scenario Sensitivity Matrix
        </button>

        <button
          onClick={() => setSelectedTab("integrity")}
          style={{
            padding: "8px 18px",
            borderRadius: "var(--radius-md)",
            fontSize: "0.88rem",
            fontWeight: 600,
            cursor: "pointer",
            background: selectedTab === "integrity" ? "var(--accent-cyan-bg)" : "transparent",
            color: selectedTab === "integrity" ? "var(--accent-cyan)" : "var(--text-secondary)",
            border: selectedTab === "integrity" ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
          }}
        >
          Data Integrity & Investigation
        </button>
      </div>

      {/* TAB 1: POLICIES & BASELINES */}
      {selectedTab === "policies" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
          {/* Key Findings Card */}
          <div className="glass-card" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px" }}>
            <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--risk-low)", marginBottom: "8px" }}>
                <DollarSign size={20} />
                <span style={{ fontSize: "0.8rem", fontWeight: 700, textTransform: "uppercase" }}>Net Financial Savings</span>
              </div>
              <span style={{ fontSize: "1.8rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                $649,433.00
              </span>
              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "4px" }}>
                Candidate Policy B vs Accept-All baseline (80.0% fraud cost reduction on held-out test partition).
              </p>
            </div>

            <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--accent-cyan)", marginBottom: "8px" }}>
                <ShieldCheck size={20} />
                <span style={{ fontSize: "0.8rem", fontWeight: 700, textTransform: "uppercase" }}>Analyst Precision Lift</span>
              </div>
              <span style={{ fontSize: "1.8rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                51.01%
              </span>
              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "4px" }}>
                1 in every 1.96 flagged transactions is confirmed fraud (vs. 10.43% under Logistic Regression).
              </p>
            </div>

            <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--accent-purple)", marginBottom: "8px" }}>
                <TrendingUp size={20} />
                <span style={{ fontSize: "0.8rem", fontWeight: 700, textTransform: "uppercase" }}>Caseload Reduction</span>
              </div>
              <span style={{ fontSize: "1.8rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                -83.5%
              </span>
              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "4px" }}>
                Manual review volume slashed from 26,089 alerts (Logistic Regression) to 4,297 alerts (Policy B).
              </p>
            </div>
          </div>

          {/* Headline Model Benchmarks with Bootstrap CIs */}
          <div className="glass-card">
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Head-to-Head Model Benchmarks (1,000 Bootstrap CIs)</h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                Evaluated on held-out test partition (N = 118,108, 4,064 frauds, TransactionDT &gt; 12,192,854)
              </p>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)" }}>
                    <th style={{ padding: "12px 16px" }}>Model Candidate</th>
                    <th style={{ padding: "12px 16px" }}>PR-AUC (Primary) [95% CI]</th>
                    <th style={{ padding: "12px 16px" }}>ROC-AUC [95% CI]</th>
                    <th style={{ padding: "12px 16px" }}>Recall @ 1% FPR</th>
                    <th style={{ padding: "12px 16px" }}>Recall @ 5% FPR</th>
                    <th style={{ padding: "12px 16px" }}>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(56, 189, 248, 0.04)" }}>
                    <td style={{ padding: "14px 16px", fontWeight: 700, color: "var(--accent-cyan)" }}>
                      Champion LightGBM (scale_pos_weight=27.46)
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      0.5441 <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[0.5282, 0.5607]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>
                      0.9035 <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[0.8982, 0.9087]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 700 }}>
                      46.63% <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[44.90%, 48.24%]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>
                      65.95% <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[64.46%, 67.45%]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.0246</td>
                  </tr>

                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "14px 16px", fontWeight: 600 }}>
                      Baseline Logistic Regression (StandardScaler + Balanced)
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>
                      0.2746 <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[0.2605, 0.2891]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>
                      0.8092 <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>[0.8021, 0.8164]</span>
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>15.08%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>41.76%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.1782</td>
                  </tr>

                  <tr>
                    <td style={{ padding: "14px 16px", color: "var(--risk-low)", fontWeight: 700 }}>
                      Relative Performance Lift
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 700 }}>
                      +98.1% Lift
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
                      +11.7% Lift
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 700 }}>
                      3.09x Capture
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
                      +24.19 pp
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
                      -86.2% Error
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Candidate Policy Comparison Table */}
          <div className="glass-card">
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>3-Tier Operating Policy Comparison Table</h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                Economic model: Total Cost = (FN_low + 0.2*FN_med)*$200 + Reviews*$8.00 + Challenges*$0.50
              </p>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)" }}>
                    <th style={{ padding: "12px 16px" }}>Policy Name</th>
                    <th style={{ padding: "12px 16px" }}>Operating Cutoffs</th>
                    <th style={{ padding: "12px 16px" }}>Review Rate % (Vol)</th>
                    <th style={{ padding: "12px 16px" }}>High-Tier Recall</th>
                    <th style={{ padding: "12px 16px" }}>Precision</th>
                    <th style={{ padding: "12px 16px" }}>FPR</th>
                    <th style={{ padding: "12px 16px" }}>Total Expected Cost</th>
                    <th style={{ padding: "12px 16px" }}>Net Savings</th>
                    <th style={{ padding: "12px 16px" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "14px 16px" }}>No Model: Accept All</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>N/A</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.0% (0)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.0%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>N/A</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.0%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-high)" }}>$812,800.00</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$0.00</td>
                    <td style={{ padding: "14px 16px" }}><span className="badge badge-high">Rejected</span></td>
                  </tr>

                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "14px 16px" }}>Naive Amount Rule (&gt; $500)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>Amt &gt; 500</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>4.08% (4,816)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>5.76%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-high)" }}>4.86%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>4.02%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$804,528.00</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$8,272.00</td>
                    <td style={{ padding: "14px 16px" }}><span className="badge badge-high">Rejected</span></td>
                  </tr>

                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "14px 16px" }}>Logistic Regression Baseline</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>p &gt;= 0.50</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-high)" }}>22.09% (26,089)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>66.95%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-high)" }}>10.43%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>20.49%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$477,312.00</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$335,488.00</td>
                    <td style={{ padding: "14px 16px" }}><span className="badge badge-med">Queue Overflow</span></td>
                  </tr>

                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "14px 16px", fontWeight: 600 }}>Candidate Policy A (Conservative)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.01 / 0.96</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>0.88% (1,044)</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>23.08%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", color: "var(--risk-low)", fontWeight: 700 }}>89.85%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>0.093%</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$189,169.50</td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>$623,630.50</td>
                    <td style={{ padding: "14px 16px" }}><span className="badge badge-low">1% Cap Viable</span></td>
                  </tr>

                  <tr style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid var(--risk-low-border)" }}>
                    <td style={{ padding: "14px 16px", fontWeight: 700, color: "var(--risk-low)" }}>
                      Candidate Policy B (Balanced)
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      0.01 / 0.70
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      3.64% (4,297)
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      53.94%
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      51.01%
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)" }}>
                      1.85%
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--risk-low)" }}>
                      $163,367.00
                    </td>
                    <td style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontWeight: 800, color: "var(--risk-low)" }}>
                      $649,433.00
                    </td>
                    <td style={{ padding: "14px 16px" }}>
                      <span className="badge badge-low" style={{ background: "var(--risk-low)", color: "#070a13" }}>
                        RECOMMENDED
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: 36-SCENARIO SENSITIVITY MATRIX */}
      {selectedTab === "sensitivity" && data && (
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div className="glass-card">
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>36-Scenario Financial Sensitivity Matrix</h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                Evaluating policy stability across 3 Fraud Losses ($160, $200, $240) × 3 Review Costs ($5, $8, $12) × 4 Capacity Constraints (1%, 3%, 5%, 10%)
              </p>
            </div>

            <div style={{ overflowX: "auto", maxHeight: "500px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
                <thead style={{ position: "sticky", top: 0, background: "var(--bg-card)", zIndex: 10 }}>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)" }}>
                    <th style={{ padding: "10px 14px" }}>Scenario ID</th>
                    <th style={{ padding: "10px 14px" }}>Fraud Loss</th>
                    <th style={{ padding: "10px 14px" }}>Review Cost</th>
                    <th style={{ padding: "10px 14px" }}>Capacity Cap</th>
                    <th style={{ padding: "10px 14px" }}>Optimal τ_high</th>
                    <th style={{ padding: "10px 14px" }}>Review Rate %</th>
                    <th style={{ padding: "10px 14px" }}>High Recall</th>
                    <th style={{ padding: "10px 14px" }}>Total Cost</th>
                    <th style={{ padding: "10px 14px" }}>Net Savings</th>
                  </tr>
                </thead>
                <tbody>
                  {data.financial_sensitivity_matrix_36_scenarios.map((sc, idx) => (
                    <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>{sc.scenario_id}</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>${sc.fraud_loss}</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>${sc.review_cost}</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>{sc.capacity_cap_pct}%</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>{sc.optimal_tau_high.toFixed(2)}</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>{sc.manual_review_rate_pct.toFixed(2)}%</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>{sc.recall_high_tier_pct.toFixed(2)}%</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)" }}>${sc.total_expected_cost.toLocaleString()}</td>
                      <td style={{ padding: "10px 14px", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>${sc.net_savings_vs_accept_all.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Generalization Stress Test */}
          <div className="glass-card">
            <div style={{ marginBottom: "12px" }}>
              <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Generalization Stress Test (Unseen Entities Benchmark)</h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                Evaluating model degradation on completely novel cards with 0% entity overlap (N = 10,952)
              </p>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", marginTop: "16px" }}>
              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block" }}>PR-AUC on Unseen Cards</span>
                <span style={{ fontSize: "1.5rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>0.4487</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginTop: "2px" }}>
                  -17.53% relative decay vs 0.5441 temporal benchmark
                </span>
              </div>

              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block" }}>ROC-AUC on Unseen Cards</span>
                <span style={{ fontSize: "1.5rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>0.8774</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginTop: "2px" }}>
                  -2.89% relative decay vs 0.9035 temporal benchmark
                </span>
              </div>

              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block" }}>Recall @ 1% FPR Limit</span>
                <span style={{ fontSize: "1.5rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--risk-med)" }}>36.36%</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginTop: "2px" }}>
                  -22.02% capture reduction vs 46.63% temporal benchmark
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: DATA INTEGRITY & INVESTIGATION */}
      {selectedTab === "integrity" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div className="glass-card">
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "8px" }}>Data Integrity Investigation Facts</h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "20px" }}>
              Key architectural and statistical truths discovered during the Week 1 investigation:
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <strong style={{ color: "var(--accent-cyan)" }}>Temporal Delta Verification:</strong>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  <code>TransactionDT</code> is a relative interval in seconds spanning 182 days (26 weeks). Origin is undisclosed; timestamp reconstruction is mathematically non-identifiable.
                </p>
              </div>

              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <strong style={{ color: "var(--accent-cyan)" }}>Entity Overlap & Leakage:</strong>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  Temporal split produces 67.6% entity overlap (realistic production deployment). Random split (74.7%) was rejected due to entity leakage inflating PR-AUC by +24.8%.
                </p>
              </div>

              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <strong style={{ color: "var(--accent-cyan)" }}>Feature Audit Signal Identicality:</strong>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  <code>C1</code> is identical to transaction velocity (|r|=1.000) and <code>D1</code> is identical to time-since-last-transaction (|r|=1.000). Rebuilding them is redundant.
                </p>
              </div>

              <div style={{ padding: "16px", background: "rgba(0, 0, 0, 0.2)", borderRadius: "var(--radius-md)" }}>
                <strong style={{ color: "var(--accent-cyan)" }}>V-Feature Collinearity Consolidation:</strong>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  162 V-feature pairs have |r| &gt;= 0.98. TreeSHAP attributions consolidate these into unified domain clusters to eliminate multi-collinear explanation clutter.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
