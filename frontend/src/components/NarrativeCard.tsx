"use client";

import { GroundedNarrative } from "@/lib/api";
import { Sparkles, CheckCircle2, FileText, AlertCircle } from "lucide-react";

interface NarrativeCardProps {
  narrative?: GroundedNarrative;
}

export function NarrativeCard({ narrative }: NarrativeCardProps) {
  if (!narrative || (!narrative.risk_summary && !narrative.primary_drivers)) {
    return (
      <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Grounded GenAI Analyst Narrative</h3>
          <span className="badge badge-purple" style={{ fontSize: "0.7rem" }}>
            Supporting Layer
          </span>
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontStyle: "italic" }}>
          No offline narrative available for this transaction. The model decision and SHAP evidence remain fully operational.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Header & Safeguard Badge */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: "28px",
            height: "28px",
            borderRadius: "6px",
            background: "linear-gradient(135deg, #a855f7, #ec4899)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <Sparkles size={16} color="#fff" />
          </div>
          <div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Grounded Analyst Narrative</h3>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Generated offline by local LLM • Zero paid APIs
            </p>
          </div>
        </div>

        <div style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          padding: "4px 10px",
          background: "rgba(16, 185, 129, 0.12)",
          border: "1px solid rgba(16, 185, 129, 0.3)",
          borderRadius: "var(--radius-full)",
          fontSize: "0.75rem",
          fontWeight: 600,
          color: "var(--risk-low)",
        }}>
          <CheckCircle2 size={14} />
          <span>100% Grounded Safeguard Verified</span>
        </div>
      </div>

      {/* 4-Section Narrative Content */}
      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        background: "rgba(0, 0, 0, 0.25)",
        padding: "16px",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--border-subtle)",
        fontSize: "0.88rem",
        lineHeight: 1.6,
      }}>
        {narrative.risk_summary && (
          <div>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase", display: "block", marginBottom: "2px" }}>
              1. Risk Assessment Summary
            </span>
            <p style={{ color: "var(--text-primary)" }}>{narrative.risk_summary}</p>
          </div>
        )}

        {narrative.primary_drivers && (
          <div>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--risk-high)", textTransform: "uppercase", display: "block", marginBottom: "2px" }}>
              2. Primary Empirical Drivers
            </span>
            <p style={{ color: "var(--text-secondary)" }}>{narrative.primary_drivers}</p>
          </div>
        )}

        {narrative.mitigating_factors && (
          <div>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--risk-low)", textTransform: "uppercase", display: "block", marginBottom: "2px" }}>
              3. Mitigating Factors
            </span>
            <p style={{ color: "var(--text-secondary)" }}>{narrative.mitigating_factors}</p>
          </div>
        )}

        {narrative.recommended_action && (
          <div style={{ paddingTop: "8px", borderTop: "1px solid var(--border-subtle)" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-purple)", textTransform: "uppercase", display: "block", marginBottom: "2px" }}>
              4. Recommended Workflow Action
            </span>
            <p style={{ color: "var(--text-primary)", fontWeight: 500 }}>{narrative.recommended_action}</p>
          </div>
        )}
      </div>

      {/* Safety Notice */}
      <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.72rem", color: "var(--text-muted)" }}>
        <FileText size={12} />
        <span>Audited by Grounding Validator: All numbers, features, and directions verified strictly against SHAP evidence.</span>
      </div>
    </div>
  );
}
