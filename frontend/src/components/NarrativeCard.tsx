"use client";

import { useState } from "react";
import { GroundedNarrative } from "@/lib/api";
import { Info, ChevronDown, ChevronUp, ShieldAlert, ShieldCheck, CheckCircle2 } from "lucide-react";

interface NarrativeCardProps {
  narrative?: GroundedNarrative;
}

/**
 * Parses markdown-style bullet strings into clean structured items.
 * Example input line: "- **Card Sub-Type Code**: Observed value `174.0` (+0.125 SHAP log-odds). Secondary card classification code"
 */
function renderMarkdownBullets(rawText: string, isRiskDriver: boolean) {
  if (!rawText) return null;

  const lines = rawText
    .split(/\n+/)
    .map((l) => l.trim().replace(/^[-*•]\s*/, ""))
    .filter(Boolean);

  if (lines.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      {lines.map((line, idx) => {
        const titleMatch = line.match(/^\*\*([^*]+)\*\*:\s*(.*)$/);
        const title = titleMatch ? titleMatch[1] : null;
        const rest = titleMatch ? titleMatch[2] : line;

        const cleanRest = rest
          .replace(/`([^`]+)`/g, "$1")
          .replace(/\*\*([^*]+)\*\*/g, "$1")
          .replace(/^###\s*/, "");

        return (
          <div
            key={idx}
            style={{
              padding: "8px 12px",
              background: "var(--bg-app)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              fontSize: "0.82rem",
              lineHeight: 1.5,
            }}
          >
            {title ? (
              <>
                <strong style={{ color: isRiskDriver ? "var(--risk-high)" : "var(--risk-low)", marginRight: "4px" }}>
                  {title}:
                </strong>
                <span style={{ color: "var(--text-secondary)" }}>{cleanRest}</span>
              </>
            ) : (
              <span style={{ color: "var(--text-secondary)" }}>{cleanRest}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function NarrativeCard({ narrative }: NarrativeCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (!narrative || (!narrative.risk_summary && !narrative.primary_drivers)) {
    return (
      <div className="card" style={{ padding: "16px 20px" }}>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontStyle: "italic" }}>
          No analyst summary available for this transaction.
        </p>
      </div>
    );
  }

  const cleanSummary = (narrative.risk_summary || "")
    .replace(/###\s*/g, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();

  const cleanAction = (narrative.recommended_action || "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/^####\s*Recommended Workflow:\s*/i, "")
    .trim();

  return (
    <div className="card">
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "0.875rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Analyst Summary
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
            · precomputed narrative
          </span>
        </div>

        <span className="chip chip-grey" style={{ fontSize: "0.7rem", gap: "4px" }}>
          <CheckCircle2 size={11} color="var(--risk-low)" /> Verified
        </span>
      </div>

      {/* Clean high-level summary box */}
      <div
        style={{
          background: "var(--bg-app)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "12px 14px",
          fontSize: "0.875rem",
          color: "var(--text-primary)",
          lineHeight: 1.6,
          marginBottom: "12px",
        }}
      >
        {cleanSummary}
      </div>

      {/* Expandable full assessment */}
      {(narrative.primary_drivers || narrative.mitigating_factors || cleanAction) && (
        <>
          <button
            onClick={() => setExpanded((v) => !v)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "5px",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              fontSize: "0.8rem",
              color: "var(--accent-blue)",
              fontWeight: 600,
              marginBottom: expanded ? "14px" : 0,
            }}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {expanded ? "Hide detailed evidence" : "Show detailed evidence"}
          </button>

          {expanded && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "14px",
                fontSize: "0.85rem",
                lineHeight: 1.6,
              }}
              className="fade-in"
            >
              {narrative.primary_drivers && (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                    <ShieldAlert size={14} color="var(--risk-high)" />
                    <p style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--risk-high)", textTransform: "uppercase" }}>
                      Key Risk Factors
                    </p>
                  </div>
                  {renderMarkdownBullets(narrative.primary_drivers, true)}
                </div>
              )}

              {narrative.mitigating_factors && (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                    <ShieldCheck size={14} color="var(--risk-low)" />
                    <p style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--risk-low)", textTransform: "uppercase" }}>
                      Mitigating Factors
                    </p>
                  </div>
                  {renderMarkdownBullets(narrative.mitigating_factors, false)}
                </div>
              )}

              {cleanAction && (
                <div style={{ paddingTop: "10px", borderTop: "1px solid var(--border)" }}>
                  <p style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-blue)", textTransform: "uppercase", marginBottom: "4px" }}>
                    Recommended Workflow Action
                  </p>
                  <p style={{ color: "var(--text-primary)", fontWeight: 500, fontSize: "0.85rem" }}>
                    {cleanAction}
                  </p>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Footnote */}
      <div style={{ display: "flex", alignItems: "center", gap: "5px", marginTop: "12px", fontSize: "0.7rem", color: "var(--text-muted)" }}>
        <Info size={11} />
        <span>Precomputed analyst narrative — generated offline during analysis, not at request time.</span>
      </div>
    </div>
  );
}
