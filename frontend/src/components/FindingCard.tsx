"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

interface FindingCardProps {
  finding: string;
  why: string;
  decision: string;
  tag?: string;
}

export function FindingCard({ finding, why, decision, tag }: FindingCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="card"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        height: "100%",
        justifyContent: "space-between",
      }}
    >
      <div>
        {tag && (
          <div style={{ marginBottom: "6px" }}>
            <span className="chip chip-grey" style={{ fontSize: "0.68rem" }}>
              {tag}
            </span>
          </div>
        )}
        <h3
          style={{
            fontSize: "0.875rem",
            fontWeight: 700,
            color: "var(--text-primary)",
            lineHeight: 1.4,
            marginBottom: "6px",
          }}
        >
          {finding}
        </h3>
        <p
          style={{
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            lineHeight: 1.6,
          }}
        >
          {why}
        </p>
      </div>

      <div style={{ paddingTop: "8px", borderTop: "1px solid var(--border)" }}>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            width: "100%",
            background: "none",
            border: "none",
            padding: "4px 0",
            cursor: "pointer",
            fontSize: "0.75rem",
            fontWeight: 600,
            color: "var(--accent-blue)",
          }}
        >
          <span>{expanded ? "Hide decision impact" : "What decision it changed →"}</span>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {expanded && (
          <div
            className="fade-in"
            style={{
              marginTop: "8px",
              padding: "8px 10px",
              background: "var(--bg-app)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.78rem",
              color: "var(--text-primary)",
              lineHeight: 1.5,
            }}
          >
            <strong style={{ color: "var(--text-secondary)", display: "block", marginBottom: "2px", fontSize: "0.7rem", textTransform: "uppercase" }}>
              Decision impact:
            </strong>
            {decision}
          </div>
        )}
      </div>
    </div>
  );
}
