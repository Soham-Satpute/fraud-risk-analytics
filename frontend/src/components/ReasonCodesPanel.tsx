"use client";

import { ReasonCode } from "@/lib/api";
import { TrendingUp, TrendingDown, Layers } from "lucide-react";

interface ReasonCodesPanelProps {
  reasonCodes?: ReasonCode[];
}

export function ReasonCodesPanel({ reasonCodes = [] }: ReasonCodesPanelProps) {
  const riskDrivers = reasonCodes.filter((rc) => rc.direction === "INCREASES_RISK");
  const mitigatingFactors = reasonCodes.filter((rc) => rc.direction === "REDUCES_RISK");

  // Max absolute SHAP for relative bar width
  const maxShap = Math.max(
    ...reasonCodes.map((rc) => Math.abs(rc.shap_value || 0.1)),
    1.0
  );

  return (
    <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>TreeSHAP Feature Attributions</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Aggregated reason codes with collinearity consolidation (162 pairs consolidated)
          </p>
        </div>
        <span className="badge badge-purple" style={{ fontSize: "0.7rem" }}>
          Authoritative Evidence
        </span>
      </div>

      {reasonCodes.length === 0 ? (
        <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.9rem" }}>
          No SHAP attribution records loaded for this transaction.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Primary Risk Drivers */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px" }}>
              <TrendingUp size={16} color="var(--risk-high)" />
              <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--risk-high)", textTransform: "uppercase" }}>
                Primary Risk Drivers (+SHAP)
              </span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {riskDrivers.map((rc, idx) => {
                const widthPct = Math.min(100, Math.max(10, (Math.abs(rc.shap_value) / maxShap) * 100));
                return (
                  <div
                    key={`risk-${idx}`}
                    style={{
                      padding: "10px 14px",
                      background: "rgba(0, 0, 0, 0.25)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border-subtle)",
                      display: "flex",
                      flexDirection: "column",
                      gap: "6px",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>{rc.display_name || rc.feature}</span>
                        {rc.cluster_name && (
                          <span
                            style={{
                              fontSize: "0.65rem",
                              background: "rgba(168, 85, 247, 0.15)",
                              color: "var(--accent-purple)",
                              padding: "2px 6px",
                              borderRadius: "4px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "4px",
                            }}
                          >
                            <Layers size={10} />
                            {rc.cluster_name}
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: "0.8rem", fontFamily: "var(--font-mono)", color: "var(--risk-high)" }}>
                        +{rc.shap_value.toFixed(2)}
                      </span>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      <span>Observed: <strong>{String(rc.value)}</strong> {rc.unit || ""}</span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}>raw: {rc.feature}</span>
                    </div>

                    {/* Progress Bar */}
                    <div style={{ height: "4px", width: "100%", background: "rgba(255, 255, 255, 0.05)", borderRadius: "2px", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${widthPct}%`, background: "var(--risk-high)", borderRadius: "2px" }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Mitigating Factors */}
          {mitigatingFactors.length > 0 && (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px" }}>
                <TrendingDown size={16} color="var(--risk-low)" />
                <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--risk-low)", textTransform: "uppercase" }}>
                  Mitigating Low-Risk Factors (-SHAP)
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {mitigatingFactors.map((rc, idx) => {
                  const widthPct = Math.min(100, Math.max(10, (Math.abs(rc.shap_value) / maxShap) * 100));
                  return (
                    <div
                      key={`mit-${idx}`}
                      style={{
                        padding: "10px 14px",
                        background: "rgba(0, 0, 0, 0.25)",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-subtle)",
                        display: "flex",
                        flexDirection: "column",
                        gap: "6px",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>{rc.display_name || rc.feature}</span>
                        <span style={{ fontSize: "0.8rem", fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
                          {rc.shap_value.toFixed(2)}
                        </span>
                      </div>

                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                        <span>Observed: <strong>{String(rc.value)}</strong> {rc.unit || ""}</span>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}>raw: {rc.feature}</span>
                      </div>

                      <div style={{ height: "4px", width: "100%", background: "rgba(255, 255, 255, 0.05)", borderRadius: "2px", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${widthPct}%`, background: "var(--risk-low)", borderRadius: "2px" }} />
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
