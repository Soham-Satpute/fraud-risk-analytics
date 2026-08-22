"use client";

import { useEffect, useState, useRef } from "react";
import { fetchReplayData, DemoReplayItem } from "@/lib/api";
import { ScoreGauge } from "@/components/ScoreGauge";
import { ReasonCodesPanel } from "@/components/ReasonCodesPanel";
import { NarrativeCard } from "@/components/NarrativeCard";
import { TransactionSandbox } from "@/components/TransactionSandbox";
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  CheckCircle,
  AlertCircle,
  Database,
  Layers,
} from "lucide-react";

// -----------------------------------------------------------
// Stream stats computed from the "seen" portion of the stream
// -----------------------------------------------------------
function useStreamStats(items: DemoReplayItem[], current: number) {
  const seen = items.slice(0, current + 1);
  const total = seen.length;
  const fraudCount = seen.filter((i) => i.predicted_risk_tier === "HIGH").length;
  const avgProb =
    total > 0
      ? seen.reduce((s, i) => s + (i.predicted_probability || 0), 0) / total
      : 0;
  return { total, fraudCount, avgProb };
}

export default function DemoPage() {
  const [replayItems, setReplayItems] = useState<DemoReplayItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(3000);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [customResult, setCustomResult] = useState<any>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load held-out test transactions (static replay slice)
  useEffect(() => {
    async function loadData() {
      try {
        setIsLoading(true);
        const data = await fetchReplayData();
        if (data && data.length > 0) {
          setReplayItems(data);
          setLoadError(null);
        } else {
          setLoadError("Replay transaction dataset could not be loaded.");
        }
      } catch (err) {
        console.error("Failed to load replay stream:", err);
        setLoadError("Unable to retrieve replay stream data.");
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, []);

  // Auto-play timer
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!isPlaying || replayItems.length === 0) return;
    intervalRef.current = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % replayItems.length);
      setCustomResult(null);
    }, playbackSpeed);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, replayItems, playbackSpeed]);

  const currentTx = replayItems[currentIndex];
  const stats = useStreamStats(replayItems, currentIndex);

  // Active values — custom override if sandbox was used
  const activeProb =
    customResult?.predicted_probability ?? currentTx?.predicted_probability ?? 0.05;
  const activeTier =
    customResult?.predicted_risk_tier ?? currentTx?.predicted_risk_tier ?? "LOW";
  const activeAction =
    customResult?.decision_action ?? currentTx?.decision_action ?? "APPROVE";
  const activeWorkflow =
    customResult?.recommended_workflow ??
    currentTx?.recommended_workflow ??
    "Approve transaction straight-through";

  const activeReasonCodes = customResult
    ? [
        ...(customResult.explanation?.top_risk_factors || []),
        ...(customResult.explanation?.top_mitigating_factors || []),
      ]
    : currentTx?.reason_codes ?? [];

  const activeNarrative = customResult
    ? {
        risk_summary: `Custom transaction scored at ${(activeProb * 100).toFixed(1)}% fraud risk.`,
        recommended_action: activeWorkflow,
        grounding_verified: false,
      }
    : currentTx?.grounded_narrative;

  const progressPct =
    replayItems.length > 0 ? ((currentIndex + 1) / replayItems.length) * 100 : 0;

  return (
    <div className="container" style={{ paddingTop: "28px", paddingBottom: "48px" }}>
      {/* Page header */}
      <div style={{ marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800 }}>
            Fraud Model Replay &amp; Interactive Sandbox
          </h1>
          <span className="chip chip-grey" style={{ fontSize: "0.72rem" }}>
            <Database size={11} /> 1,500 Held-Out Transactions
          </span>
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", maxWidth: "680px" }}>
          A replay of 1,500 held-out test transactions scored by the fraud model.
          Model scores, TreeSHAP reason codes, and analyst narratives are precomputed from the test evaluation partition.
        </p>
      </div>

      {/* Loading & Error State */}
      {isLoading && (
        <div
          className="card"
          style={{
            padding: "32px",
            textAlign: "center",
            marginBottom: "20px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <div
            style={{
              width: "28px",
              height: "28px",
              border: "2px solid var(--border-light)",
              borderTopColor: "var(--accent-blue)",
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
            }}
          />
          <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Loading replay dataset...
          </span>
        </div>
      )}

      {loadError && (
        <div
          className="card"
          style={{
            padding: "16px 20px",
            marginBottom: "20px",
            borderColor: "var(--risk-high-border)",
            background: "var(--risk-high-bg)",
            color: "var(--risk-high)",
            fontSize: "0.85rem",
          }}
        >
          {loadError}
        </div>
      )}

      {/* Playback controls */}
      <div
        className="card"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap",
          padding: "14px 18px",
          marginBottom: "16px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button
            type="button"
            className="btn"
            style={{ padding: "6px 10px" }}
            disabled={currentIndex === 0 || isLoading}
            onClick={() => {
              setCurrentIndex((p) => Math.max(0, p - 1));
              setCustomResult(null);
            }}
            title="Previous Transaction"
          >
            <SkipBack size={15} />
          </button>

          <button
            type="button"
            className="btn"
            style={{
              padding: "6px 14px",
              background: isPlaying ? "var(--risk-high-bg)" : "var(--accent-blue)",
              borderColor: isPlaying ? "var(--risk-high-border)" : "var(--accent-blue)",
              color: isPlaying ? "var(--risk-high)" : "#fff",
              fontWeight: 600,
            }}
            disabled={isLoading || replayItems.length === 0}
            onClick={() => setIsPlaying((p) => !p)}
          >
            {isPlaying ? <Pause size={15} /> : <Play size={15} />}
            {isPlaying ? "Pause" : "Play replay stream"}
          </button>

          <button
            type="button"
            className="btn"
            style={{ padding: "6px 10px" }}
            disabled={isLoading || replayItems.length === 0}
            onClick={() => {
              setCurrentIndex((p) => (p + 1) % (replayItems.length || 1));
              setCustomResult(null);
            }}
            title="Next Transaction"
          >
            <SkipForward size={15} />
          </button>
        </div>

        {/* Speed selector */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            borderLeft: "1px solid var(--border)",
            paddingLeft: "12px",
          }}
        >
          <label style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>Speed:</label>
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            className="input"
            style={{ width: "auto", padding: "4px 8px", fontSize: "0.78rem" }}
          >
            <option value={5000}>Slow (5s)</option>
            <option value={3000}>Normal (3s)</option>
            <option value={1500}>Fast (1.5s)</option>
          </select>
        </div>

        {/* Mode indicator */}
        {customResult ? (
          <span className="chip chip-blue" style={{ fontSize: "0.75rem" }}>
            <Layers size={12} /> Custom Sandbox Transaction
          </span>
        ) : (
          <span className="chip chip-grey" style={{ fontSize: "0.75rem" }}>
            Precomputed Replay
          </span>
        )}

        {/* Progress indicator */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            {isLoading
              ? "Loading..."
              : `Transaction ${currentIndex + 1} of ${replayItems.length}`}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ marginBottom: "20px" }}>
        <div
          style={{
            height: "4px",
            background: "var(--bg-surface)",
            borderRadius: "2px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progressPct}%`,
              background: "var(--accent-blue)",
              borderRadius: "2px",
              transition: "width 0.4s ease",
            }}
          />
        </div>
      </div>

      {/* Transaction info bar */}
      {currentTx && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
            padding: "12px 16px",
            background: "var(--bg-surface)",
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border)",
            marginBottom: "20px",
            fontSize: "0.85rem",
          }}
        >
          <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
            <span>
              <span style={{ color: "var(--text-muted)" }}>Amount: </span>
              <strong style={{ fontFamily: "var(--font-mono)" }}>
                ${Number(currentTx.TransactionAmt).toFixed(2)}
              </strong>
            </span>
            <span>
              <span style={{ color: "var(--text-muted)" }}>Type: </span>
              <strong>{currentTx.ProductCD || "W"}</strong>
            </span>
            <span>
              <span style={{ color: "var(--text-muted)" }}>Card Network: </span>
              <strong>{currentTx.card4 ? `${currentTx.card4} (${currentTx.card6 || "card"})` : "—"}</strong>
            </span>
            <span>
              <span style={{ color: "var(--text-muted)" }}>Email: </span>
              <strong>{currentTx.P_emaildomain || "—"}</strong>
            </span>
            <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
              ID #{currentTx.TransactionID}
            </span>
          </div>

          {/* Ground truth label */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Actual outcome:</span>
            {currentTx.isFraud === 1 ? (
              <span className="chip chip-high" style={{ gap: "4px" }}>
                <AlertCircle size={12} /> Fraud confirmed
              </span>
            ) : (
              <span className="chip chip-low" style={{ gap: "4px" }}>
                <CheckCircle size={12} /> Legitimate
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main 2-column grid */}
      <div className="grid-2">
        {/* Left: Score + Sandbox */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <ScoreGauge
            probability={activeProb}
            riskTier={activeTier}
            decisionAction={activeAction}
            workflow={activeWorkflow}
            latencyMs={313.5}
          />
          <TransactionSandbox onScoreResult={(res) => setCustomResult(res)} />
        </div>

        {/* Right: Reason Codes + Analyst note */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <ReasonCodesPanel reasonCodes={activeReasonCodes} />
          <NarrativeCard narrative={activeNarrative} />
        </div>
      </div>

      {/* Stream stats bar */}
      <div
        style={{
          marginTop: "24px",
          display: "flex",
          gap: "0",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          overflow: "hidden",
          background: "var(--bg-surface)",
        }}
      >
        {[
          { label: "Transactions seen in replay", value: stats.total.toLocaleString() },
          { label: "High-risk flagged (≥ 70%)", value: stats.fraudCount.toLocaleString() },
          { label: "Avg replay risk score", value: `${(stats.avgProb * 100).toFixed(1)}%` },
        ].map((s, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              textAlign: "center",
              padding: "14px 16px",
              borderRight: i < 2 ? "1px solid var(--border)" : "none",
            }}
          >
            <div
              style={{
                fontSize: "1.2rem",
                fontWeight: 800,
                fontFamily: "var(--font-mono)",
                color: "var(--text-primary)",
              }}
            >
              {s.value}
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
