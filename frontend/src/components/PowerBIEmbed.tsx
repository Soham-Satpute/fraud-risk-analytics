"use client";

import { useState, useEffect } from "react";
import {
  BarChart3,
  Maximize2,
  FileText,
  ExternalLink,
  X,
  TrendingUp,
  ShieldCheck,
  Activity,
  Layers,
  Table,
  CheckCircle2,
  DollarSign,
  ZoomIn,
} from "lucide-react";

interface PowerBIEmbedProps {
  imageSrc?: string;
  pdfUrl?: string;
}

export function PowerBIEmbed({
  imageSrc = "/dashboard-powerbi.png",
  pdfUrl = "/Fraud-risk-analytics.pdf",
}: PowerBIEmbedProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<number>(0);

  // Close modal on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsModalOpen(false);
      }
    };
    if (isModalOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "unset";
    };
  }, [isModalOpen]);

  const dashboardFeatures = [
    {
      id: "kpis",
      title: "Executive KPI Ribbon",
      icon: Activity,
      summary: "Real-time batch telemetry across held-out transactions.",
      details:
        "Tracks 2,000 scored transactions with a 21.80% operational alert rate (Medium + High tiers), 54.13% precision, 0.23 average fraud probability, $22,626 – $42,615 estimated batch cost savings, and verified Normal Stability health.",
    },
    {
      id: "trend",
      title: "Alert Rate Trend Over Time",
      icon: TrendingUp,
      summary: "Daily alert rate volatility tracking (8.6% – 38.7%).",
      details:
        "Plots alert percentages across 36 consecutive test periods (days 141 to 176). Highlights stable alert frequency and prevents runaway false alarm spikes without missing batch attack bursts.",
    },
    {
      id: "tiers",
      title: "3-Tier Risk Distribution",
      icon: Layers,
      summary: "Straight-Through vs Step-Up vs Manual Review.",
      details:
        "Donut distribution showing 712 Low Risk (47.47% straight-through approved), 461 Medium Risk (30.73% step-up challenged), and 327 High Risk (21.80% manual review queue).",
    },
    {
      id: "gauges",
      title: "Precision & Recall Gauges",
      icon: ShieldCheck,
      summary: "Actual performance vs target benchmark SLAs.",
      details:
        "Precision gauge reaches 54.13% (exceeding the 46.60% target) and Recall gauge achieves 78.67% (exceeding the 65.95% target) on held-out test transactions.",
    },
    {
      id: "costs",
      title: "Cost vs. Loss Prevented & Calibration",
      icon: DollarSign,
      summary: "$35K gross fraud loss prevented vs $3K review cost.",
      details:
        "Demonstrates steep positive net savings ($33K net point estimate) and a monotonically calibrated fraud rate across score deciles (reaching peak concentration in the 90–100% bucket).",
    },
    {
      id: "queue",
      title: "High-Risk Review Queue (SHAP)",
      icon: Table,
      summary: "Triage feed with explainable feature attribution.",
      details:
        "Live transaction queue showing Transaction IDs, amounts, fraud probabilities (0.99–1.00), and top SHAP risk features such as Adversarial Risk Pattern Cluster, Transaction Velocity (C1/C13/C14), and Card Issuer Proxies.",
    },
  ];

  return (
    <div
      className="card"
      style={{
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "18px",
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "12px",
          paddingBottom: "14px",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "var(--radius-md)",
              background: "rgba(242, 200, 17, 0.15)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <BarChart3 size={18} color="#F2C811" />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)" }}>
                Power BI Executive Dashboard
              </span>
              <span className="chip chip-green" style={{ fontSize: "0.68rem", padding: "2px 6px" }}>
                🟢 Normal Stability
              </span>
            </div>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              High-Fidelity Power BI Desktop Report • Held-Out Test Replay Slice
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setIsModalOpen(true)}
            style={{
              fontSize: "0.78rem",
              padding: "6px 12px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <Maximize2 size={13} />
            <span>Inspect 4K Report</span>
          </button>

          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
            style={{
              fontSize: "0.78rem",
              padding: "6px 12px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              textDecoration: "none",
            }}
          >
            <FileText size={13} />
            <span>Open PDF File</span>
            <ExternalLink size={11} opacity={0.7} />
          </a>
        </div>
      </div>

      {/* Primary Dashboard Image Viewport */}
      <div
        style={{
          position: "relative",
          width: "100%",
          borderRadius: "var(--radius-md)",
          overflow: "hidden",
          border: "1px solid var(--border)",
          background: "#0D1117",
          cursor: "pointer",
          boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
        }}
        onClick={() => setIsModalOpen(true)}
      >
        <img
          src={imageSrc}
          alt="Power BI Fraud Risk Analytics Executive Dashboard"
          style={{
            width: "100%",
            height: "auto",
            display: "block",
            transition: "transform 0.3s ease",
          }}
          className="dashboard-preview-img"
        />

        {/* Hover zoom overlay badge */}
        <div
          style={{
            position: "absolute",
            bottom: "16px",
            right: "16px",
            background: "rgba(13, 17, 23, 0.85)",
            backdropFilter: "blur(8px)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            padding: "6px 12px",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "0.75rem",
            color: "var(--text-primary)",
            fontWeight: 600,
          }}
        >
          <ZoomIn size={14} color="var(--accent-blue)" />
          <span>Click to Zoom &amp; Inspect (4428 × 2538)</span>
        </div>
      </div>

      {/* Metric Highlights Strip matching Dashboard */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: "10px",
          marginTop: "4px",
        }}
      >
        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Total Scored
          </div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            2K (1,500)
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Held-out replay stream</div>
        </div>

        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Alert Rate
          </div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--accent-blue)" }}>
            21.80%
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Medium + High Tiers</div>
        </div>

        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Model Precision
          </div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
            54.13%
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Target SLA: 46.60%</div>
        </div>

        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Model Recall
          </div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--risk-low)" }}>
            78.67%
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Target SLA: 65.95%</div>
        </div>

        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Est. Cost Saved
          </div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--accent-blue)" }}>
            $22.6K – $42.6K
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>$649K on full portfolio</div>
        </div>

        <div className="card" style={{ padding: "10px 14px", background: "var(--bg-app)" }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
            Model Health
          </div>
          <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--risk-low)", display: "flex", alignItems: "center", gap: "4px" }}>
            <CheckCircle2 size={16} color="var(--risk-low)" /> Normal
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Drift PSI &lt; 0.10</div>
        </div>
      </div>

      {/* Interactive Visual Zones Tour */}
      <div style={{ marginTop: "6px" }}>
        <div style={{ fontSize: "0.82rem", fontWeight: 700, marginBottom: "10px", color: "var(--text-primary)" }}>
          Dashboard Architecture &amp; Component Breakdown:
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "12px",
          }}
        >
          {dashboardFeatures.map((feat, idx) => {
            const Icon = feat.icon;
            const isSelected = activeTab === idx;
            return (
              <div
                key={feat.id}
                onClick={() => setActiveTab(idx)}
                style={{
                  padding: "14px",
                  borderRadius: "var(--radius-md)",
                  border: isSelected
                    ? "1px solid var(--accent-blue)"
                    : "1px solid var(--border)",
                  background: isSelected ? "var(--accent-blue-bg)" : "var(--bg-surface)",
                  cursor: "pointer",
                  transition: "all var(--transition)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                  <Icon size={15} color={isSelected ? "var(--accent-blue)" : "var(--text-muted)"} />
                  <strong style={{ fontSize: "0.82rem", color: "var(--text-primary)" }}>
                    {feat.title}
                  </strong>
                </div>
                <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5, margin: 0 }}>
                  {feat.details}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Fullscreen Lightbox Modal */}
      {isModalOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            background: "rgba(5, 8, 14, 0.92)",
            backdropFilter: "blur(10px)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "24px",
          }}
          onClick={() => setIsModalOpen(false)}
        >
          {/* Modal Header */}
          <div
            style={{
              position: "absolute",
              top: "16px",
              left: "24px",
              right: "24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              zIndex: 1010,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontSize: "1rem", fontWeight: 700, color: "#fff" }}>
                Power BI Desktop Dashboard — High-Resolution Inspection
              </span>
              <span className="chip chip-grey" style={{ fontSize: "0.72rem" }}>
                4428 × 2538 px
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <a
                href={pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary"
                style={{ fontSize: "0.8rem", padding: "6px 14px", textDecoration: "none" }}
              >
                <FileText size={14} /> Open Original PDF
              </a>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setIsModalOpen(false)}
                style={{
                  fontSize: "0.8rem",
                  padding: "6px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <X size={16} /> Close (Esc)
              </button>
            </div>
          </div>

          {/* Modal Content */}
          <div
            style={{
              maxWidth: "96vw",
              maxHeight: "88vh",
              overflow: "auto",
              borderRadius: "var(--radius-lg)",
              border: "1px solid var(--border)",
              boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
              marginTop: "40px",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={imageSrc}
              alt="Power BI Dashboard High Resolution"
              style={{
                width: "100%",
                height: "auto",
                display: "block",
              }}
            />
          </div>
        </div>
      )}

      <style jsx>{`
        .dashboard-preview-img:hover {
          transform: scale(1.008);
        }
      `}</style>
    </div>
  );
}
