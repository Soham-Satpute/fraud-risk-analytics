"use client";

import { useState } from "react";
import { ReasonCode } from "@/lib/api";
import { ChevronDown, ChevronUp } from "lucide-react";

interface ReasonCodesPanelProps {
  reasonCodes?: ReasonCode[];
}

function buildSentence(rc: ReasonCode): string {
  const dir = rc.direction === "INCREASES_RISK" || rc.direction === "REDUCES_RISK"
    ? (rc.direction === "INCREASES_RISK" ? "raised" : "lowered")
    : "affected";

  // Use description if it exists (already plain English from the API)
  if (rc.description && rc.description.length > 8) {
    return rc.description;
  }

  const val = rc.feature_value ?? rc.value;
  const valStr = val !== undefined && val !== null && String(val).trim() !== ""
    ? ` (observed: ${String(val)}${rc.unit ? " " + rc.unit : ""})`
    : "";

  return `${rc.display_name || rc.feature}${valStr} — this ${dir} the fraud suspicion score.`;
}

export function ReasonCodesPanel({ reasonCodes = [] }: ReasonCodesPanelProps) {
  const [open, setOpen] = useState(true);

  const riskDrivers = reasonCodes.filter(
    (rc) => rc.direction === "INCREASES_RISK"
  );
  const mitigating = reasonCodes.filter(
    (rc) => rc.direction === "REDUCES_RISK" || rc.direction === "DECREASES_RISK"
  );

  const maxShap = Math.max(...reasonCodes.map((rc) => Math.abs(rc.shap_value || 0.1)), 1.0);

  const totalCount = riskDrivers.length + mitigating.length;

  return (
    <div className="card">
      {/* Collapsible header */}
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex",
          width: "100%",
          justifyContent: "space-between",
          alignItems: "center",
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: 0,
          marginBottom: open ? "16px" : 0,
        }}
      >
        <div>
          <span style={{ fontSize: "0.875rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Why this decision?
          </span>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: "8px" }}>
            {totalCount > 0 ? `${totalCount} reason${totalCount !== 1 ? "s" : ""}` : ""}
          </span>
        </div>
        {open ? (
          <ChevronUp size={16} color="var(--text-muted)" />
        ) : (
          <ChevronDown size={16} color="var(--text-muted)" />
        )}
      </button>

      {open && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {reasonCodes.length === 0 && (
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontStyle: "italic" }}>
              No reason codes available for this transaction.
            </p>
          )}

          {/* Risk-raising factors */}
          {riskDrivers.length > 0 && (
            <div>
              <p style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--risk-high)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
                What raised suspicion
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {riskDrivers.map((rc, i) => {
                  const barWidth = Math.max(8, (Math.abs(rc.shap_value) / maxShap) * 100);
                  return (
                    <div
                      key={`risk-${i}`}
                      style={{
                        padding: "10px 12px",
                        background: "var(--bg-app)",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <p style={{ fontSize: "0.83rem", color: "var(--text-primary)", marginBottom: "6px", lineHeight: 1.5 }}>
                        {buildSentence(rc)}
                      </p>
                      {/* Bar */}
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <div style={{ flex: 1, height: "4px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", overflow: "hidden" }}>
                          <div style={{ width: `${barWidth}%`, height: "100%", background: "var(--risk-high)", borderRadius: "2px" }} />
                        </div>
                        <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)", whiteSpace: "nowrap" }}>
                          strength {rc.shap_value.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Mitigating factors */}
          {mitigating.length > 0 && (
            <div>
              <p style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--risk-low)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "8px" }}>
                What worked in its favour
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {mitigating.map((rc, i) => {
                  const barWidth = Math.max(8, (Math.abs(rc.shap_value) / maxShap) * 100);
                  return (
                    <div
                      key={`mit-${i}`}
                      style={{
                        padding: "10px 12px",
                        background: "var(--bg-app)",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <p style={{ fontSize: "0.83rem", color: "var(--text-primary)", marginBottom: "6px", lineHeight: 1.5 }}>
                        {buildSentence(rc)}
                      </p>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <div style={{ flex: 1, height: "4px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", overflow: "hidden" }}>
                          <div style={{ width: `${barWidth}%`, height: "100%", background: "var(--risk-low)", borderRadius: "2px" }} />
                        </div>
                        <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)", whiteSpace: "nowrap" }}>
                          strength {Math.abs(rc.shap_value).toFixed(2)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
