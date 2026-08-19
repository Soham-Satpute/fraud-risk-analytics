"use client";

import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";

interface ScoreGaugeProps {
  probability: number;
  riskTier: "LOW" | "MEDIUM" | "HIGH";
  decisionAction: "APPROVE" | "STEP_UP_AUTH" | "MANUAL_REVIEW";
  workflow: string;
  latencyMs?: number;
}

export function ScoreGauge({
  probability,
  riskTier,
  decisionAction,
  workflow,
  latencyMs = 313,
}: ScoreGaugeProps) {
  // SVG Gauge calculations
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - probability * circumference;

  const color =
    riskTier === "HIGH"
      ? "var(--risk-high)"
      : riskTier === "MEDIUM"
      ? "var(--risk-med)"
      : "var(--risk-low)";

  const glowShadow =
    riskTier === "HIGH"
      ? "var(--shadow-glow-red)"
      : riskTier === "MEDIUM"
      ? "0 0 20px rgba(245, 158, 11, 0.25)"
      : "var(--shadow-glow-emerald)";

  const Icon =
    riskTier === "HIGH" ? ShieldAlert : riskTier === "MEDIUM" ? AlertTriangle : ShieldCheck;

  return (
    <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Risk Decision & Scoring</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Champion LightGBM model score under Candidate Policy B (τ_high=0.70, τ_med=0.01)
          </p>
        </div>
        <div style={{ textAlign: "right" }}>
          <span className="badge badge-cyan" style={{ fontSize: "0.7rem" }}>
            {latencyMs.toFixed(1)} ms inference
          </span>
        </div>
      </div>

      {/* SVG Probability Dial */}
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", position: "relative", padding: "10px 0" }}>
        <svg width="200" height="200" style={{ transform: "rotate(-90deg)" }}>
          {/* Background Ring */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="14"
            fill="transparent"
          />
          {/* Animated Value Ring */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            stroke={color}
            strokeWidth="14"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            style={{
              transition: "stroke-dashoffset 0.8s ease-in-out, stroke 0.4s ease",
              filter: `drop-shadow(${glowShadow})`,
            }}
          />
        </svg>

        {/* Center Content */}
        <div style={{
          position: "absolute",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <Icon size={32} color={color} style={{ marginBottom: "4px" }} />
          <span style={{
            fontSize: "2.2rem",
            fontWeight: 800,
            fontFamily: "var(--font-mono)",
            letterSpacing: "-0.04em",
            color: "var(--text-primary)",
          }}>
            {(probability * 100).toFixed(1)}%
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Fraud Probability
          </span>
        </div>
      </div>

      {/* Operational Directives */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "12px",
        padding: "12px",
        background: "rgba(0, 0, 0, 0.25)",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--border-subtle)",
      }}>
        <div>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>
            Assigned Risk Tier
          </span>
          <span
            className={
              riskTier === "HIGH"
                ? "badge badge-high"
                : riskTier === "MEDIUM"
                ? "badge badge-med"
                : "badge badge-low"
            }
            style={{ marginTop: "4px" }}
          >
            {riskTier} RISK
          </span>
        </div>

        <div>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block" }}>
            Operational Action
          </span>
          <span
            className={
              decisionAction === "MANUAL_REVIEW"
                ? "badge badge-high"
                : decisionAction === "STEP_UP_AUTH"
                ? "badge badge-med"
                : "badge badge-low"
            }
            style={{ marginTop: "4px" }}
          >
            {decisionAction.replace("_", " ")}
          </span>
        </div>
      </div>

      {/* Workflow Guidance */}
      <div style={{
        padding: "12px 16px",
        background:
          riskTier === "HIGH"
            ? "var(--risk-high-bg)"
            : riskTier === "MEDIUM"
            ? "var(--risk-med-bg)"
            : "var(--risk-low-bg)",
        border: `1px solid ${color}`,
        borderRadius: "var(--radius-md)",
        fontSize: "0.85rem",
        color: "var(--text-primary)",
        display: "flex",
        alignItems: "center",
        gap: "10px",
      }}>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: color }} />
        <span><strong>Policy Directive:</strong> {workflow}</span>
      </div>
    </div>
  );
}
