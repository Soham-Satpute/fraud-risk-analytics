"use client";

import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

interface ScoreGaugeProps {
  probability: number;
  riskTier: "LOW" | "MEDIUM" | "HIGH";
  decisionAction: "APPROVE" | "STEP_UP_AUTH" | "MANUAL_REVIEW";
  workflow: string;
  latencyMs?: number;
}

const TIER_CONFIG = {
  LOW: {
    color: "var(--risk-low)",
    bg: "var(--risk-low-bg)",
    border: "var(--risk-low-border)",
    icon: CheckCircle,
    label: "Low Risk",
    chipClass: "chip chip-low",
  },
  MEDIUM: {
    color: "var(--risk-med)",
    bg: "var(--risk-med-bg)",
    border: "var(--risk-med-border)",
    icon: AlertTriangle,
    label: "Medium Risk",
    chipClass: "chip chip-med",
  },
  HIGH: {
    color: "var(--risk-high)",
    bg: "var(--risk-high-bg)",
    border: "var(--risk-high-border)",
    icon: XCircle,
    label: "High Risk",
    chipClass: "chip chip-high",
  },
};

const ACTION_TEXT: Record<string, { headline: string; detail: string }> = {
  APPROVE: {
    headline: "Approved automatically",
    detail: "No friction for the customer — transaction goes through instantly.",
  },
  STEP_UP_AUTH: {
    headline: "Extra verification requested",
    detail: "Customer is asked to confirm via a one-time SMS or email code.",
  },
  MANUAL_REVIEW: {
    headline: "Sent for human review",
    detail: "A fraud analyst will investigate this transaction before it proceeds.",
  },
};

export function ScoreGauge({
  probability,
  riskTier,
  decisionAction,
  latencyMs = 313,
}: ScoreGaugeProps) {
  const cfg = TIER_CONFIG[riskTier];
  const Icon = cfg.icon;
  const action = ACTION_TEXT[decisionAction] ?? ACTION_TEXT.APPROVE;
  const pct = Math.round(probability * 100);

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Model Decision
        </span>
        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {latencyMs.toFixed(0)} ms
        </span>
      </div>

      {/* Big score number */}
      <div style={{ textAlign: "center", padding: "8px 0" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "10px", marginBottom: "6px" }}>
          <Icon size={28} color={cfg.color} />
          <span
            style={{
              fontSize: "3rem",
              fontWeight: 800,
              fontFamily: "var(--font-mono)",
              color: cfg.color,
              lineHeight: 1,
            }}
          >
            {pct}%
          </span>
        </div>
        <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
          chance this transaction is fraudulent
        </p>
      </div>

      {/* Progress bar */}
      <div>
        <div
          style={{
            height: "8px",
            background: "rgba(255,255,255,0.06)",
            borderRadius: "var(--radius-full)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${pct}%`,
              background: cfg.color,
              borderRadius: "var(--radius-full)",
              transition: "width 0.6s ease",
            }}
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.7rem",
            color: "var(--text-muted)",
            marginTop: "4px",
          }}
        >
          <span>0% (Safe)</span>
          <span>100% (Fraud)</span>
        </div>
      </div>

      {/* Risk tier chip */}
      <div style={{ display: "flex", justifyContent: "center" }}>
        <span className={cfg.chipClass} style={{ fontSize: "0.8rem", padding: "5px 14px" }}>
          <Icon size={13} />
          {cfg.label}
        </span>
      </div>

      {/* What happens next */}
      <div
        style={{
          background: cfg.bg,
          border: `1px solid ${cfg.border}`,
          borderRadius: "var(--radius-md)",
          padding: "14px 16px",
        }}
      >
        <p style={{ fontSize: "0.75rem", fontWeight: 700, color: cfg.color, textTransform: "uppercase", marginBottom: "4px" }}>
          What happens next?
        </p>
        <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "2px" }}>
          {action.headline}
        </p>
        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
          {action.detail}
        </p>
      </div>
    </div>
  );
}
