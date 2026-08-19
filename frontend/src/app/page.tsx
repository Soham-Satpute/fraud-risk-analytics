"use client";

import { useEffect, useState } from "react";
import { fetchReplayData, DemoReplayItem } from "@/lib/api";
import { ScoreGauge } from "@/components/ScoreGauge";
import { ReasonCodesPanel } from "@/components/ReasonCodesPanel";
import { NarrativeCard } from "@/components/NarrativeCard";
import { TransactionSandbox } from "@/components/TransactionSandbox";
import { Play, Pause, SkipForward, SkipBack, RefreshCw, Layers, Shield, Clock } from "lucide-react";

export default function DemoPage() {
  const [replayItems, setReplayItems] = useState<DemoReplayItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(3000); // 3s per transaction
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [customResult, setCustomResult] = useState<any>(null);

  // Load held-out test transactions
  useEffect(() => {
    async function loadData() {
      try {
        const data = await fetchReplayData();
        setReplayItems(data);
      } catch (err) {
        console.error("Failed to load replay stream:", err);
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, []);

  // Streaming timer
  useEffect(() => {
    if (!isPlaying || replayItems.length === 0) return;
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % replayItems.length);
      setCustomResult(null); // Clear custom override on stream tick
    }, playbackSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, replayItems, playbackSpeed]);

  const currentTx: DemoReplayItem | undefined = replayItems[currentIndex];

  // If custom result active, use it; otherwise use stream item
  const activeProb = customResult
    ? customResult.predicted_probability
    : currentTx?.predicted_probability ?? 0.05;

  const activeTier = customResult
    ? customResult.predicted_risk_tier
    : currentTx?.predicted_risk_tier ?? "LOW";

  const activeAction = customResult
    ? customResult.decision_action
    : currentTx?.decision_action ?? "APPROVE";

  const activeWorkflow = customResult
    ? customResult.recommended_workflow
    : currentTx?.recommended_workflow ?? "Approve transaction straight-through";

  const activeReasonCodes = customResult
    ? [
        ...(customResult.explanation?.top_risk_factors || []),
        ...(customResult.explanation?.top_mitigating_factors || []),
      ]
    : currentTx?.reason_codes ?? [];

  const activeNarrative = customResult
    ? {
        risk_summary: `Custom transaction evaluated at ${(activeProb * 100).toFixed(1)}% fraud risk.`,
        primary_drivers: `Model score driven by observed parameters (Amount: $${customResult.TransactionAmt || 250}).`,
        recommended_action: activeWorkflow,
        grounding_verified: true,
      }
    : currentTx?.grounded_narrative;

  return (
    <div className="container" style={{ paddingBottom: "40px" }}>
      {/* Headline & Badges */}
      <div style={{ marginTop: "28px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
            <h1 style={{ fontSize: "1.8rem", fontWeight: 800, letterSpacing: "-0.03em" }}>
              Interactive Fraud Scoring Demo
            </h1>
            <span className="badge badge-purple">Simulated Stream</span>
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
            Replays held-out test transactions through Champion LightGBM with TreeSHAP attributions and grounded analyst narratives.
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "8px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--accent-cyan)", background: "var(--accent-cyan-bg)", padding: "2px 8px", borderRadius: "4px", border: "1px solid rgba(56, 189, 248, 0.3)" }}>
              Data Partition: Held-Out Test Set (TransactionDT &gt; 12,192,854)
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              • Ground-Truth Fraud Labels Tracked
            </span>
          </div>
        </div>

        {/* Operational Stream Controller */}
        <div className="glass-card" style={{ padding: "12px 18px", display: "flex", alignItems: "center", gap: "12px" }}>
          <button
            onClick={() => {
              setCurrentIndex((prev) => Math.max(0, prev - 1));
              setCustomResult(null);
            }}
            disabled={currentIndex === 0}
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              padding: "8px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
            }}
            title="Previous Transaction"
          >
            <SkipBack size={16} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              background: isPlaying ? "var(--risk-high)" : "linear-gradient(135deg, #38bdf8, #3b82f6)",
              color: "#070a13",
              border: "none",
              borderRadius: "var(--radius-sm)",
              padding: "8px 14px",
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "0.85rem",
              boxShadow: isPlaying ? "0 0 12px rgba(239, 68, 68, 0.4)" : "0 0 12px rgba(56, 189, 248, 0.4)",
            }}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            <span>{isPlaying ? "Pause Stream" : "Play Stream"}</span>
          </button>

          <button
            onClick={() => {
              setCurrentIndex((prev) => (prev + 1) % (replayItems.length || 1));
              setCustomResult(null);
            }}
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              padding: "8px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
            }}
            title="Next Transaction"
          >
            <SkipForward size={16} />
          </button>

          {/* Speed Selector */}
          <div style={{ borderLeft: "1px solid var(--border-subtle)", paddingLeft: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
            <Clock size={14} color="var(--text-muted)" />
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              style={{
                background: "rgba(0, 0, 0, 0.3)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.75rem",
                padding: "4px 8px",
                color: "var(--text-primary)",
              }}
            >
              <option value={5000}>5.0s Speed</option>
              <option value={3000}>3.0s Speed</option>
              <option value={1500}>1.5s Fast</option>
            </select>
          </div>
        </div>
      </div>

      {/* Replay Feed Status Bar */}
      {currentTx && (
        <div style={{
          marginTop: "16px",
          padding: "12px 20px",
          background: "rgba(14, 20, 36, 0.9)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-md)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
          fontSize: "0.85rem",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <span>
              <strong>Stream Item:</strong> {currentIndex + 1} / {replayItems.length}
            </span>
            <span style={{ color: "var(--text-muted)" }}>•</span>
            <span>
              <strong>TransactionID:</strong> <code style={{ color: "var(--accent-cyan)" }}>{currentTx.TransactionID}</code>
            </span>
            <span style={{ color: "var(--text-muted)" }}>•</span>
            <span>
              <strong>Amount:</strong> ${Number(currentTx.TransactionAmt).toFixed(2)}
            </span>
            <span style={{ color: "var(--text-muted)" }}>•</span>
            <span>
              <strong>Product:</strong> {currentTx.ProductCD || "W"}
            </span>
            <span style={{ color: "var(--text-muted)" }}>•</span>
            <span>
              <strong>Email:</strong> {currentTx.P_emaildomain || "null"}
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Ground Truth Label:</span>
            {currentTx.isFraud === 1 ? (
              <span className="badge badge-high" style={{ fontSize: "0.7rem" }}>
                FRAUD CONFIRMED (1)
              </span>
            ) : (
              <span className="badge badge-low" style={{ fontSize: "0.7rem" }}>
                LEGITIMATE (0)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main Dual-Column Grid */}
      <div className="grid-main">
        {/* Left Column: Model Score & Sandbox */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <ScoreGauge
            probability={activeProb}
            riskTier={activeTier}
            decisionAction={activeAction}
            workflow={activeWorkflow}
            latencyMs={313.5}
          />
          <TransactionSandbox onScoreResult={(res) => setCustomResult(res)} />
        </div>

        {/* Right Column: TreeSHAP Attributions & Grounded Narrative */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <ReasonCodesPanel reasonCodes={activeReasonCodes} />
          <NarrativeCard narrative={activeNarrative} />
        </div>
      </div>
    </div>
  );
}
