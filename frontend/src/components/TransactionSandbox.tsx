"use client";

import { useState } from "react";
import { scoreTransaction } from "@/lib/api";
import { Zap, AlertCircle, RefreshCw } from "lucide-react";

interface SandboxProps {
  onScoreResult: (result: any) => void;
}

export function TransactionSandbox({ onScoreResult }: SandboxProps) {
  const [amount, setAmount] = useState<number>(250.0);
  const [productCD, setProductCD] = useState<string>("W");
  const [card1, setCard1] = useState<number>(13926);
  const [pEmail, setPEmail] = useState<string>("gmail.com");
  const [rEmail, setREmail] = useState<string>("gmail.com");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [open, setOpen] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleScore = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const payload = {
        TransactionID: Math.floor(1000000 + Math.random() * 9000000),
        TransactionAmt: amount,
        ProductCD: productCD,
        card1: card1,
        card2: 360,
        card3: 150,
        card4: "visa",
        card6: "debit",
        addr1: 315,
        addr2: 87,
        P_emaildomain: pEmail,
        R_emaildomain: rEmail,
      };
      const result = await scoreTransaction(payload);
      if (result) {
        onScoreResult(result);
      } else {
        setErrorMessage("Scoring service was unable to return a prediction. Please retry.");
      }
    } catch (err: any) {
      console.error("Sandbox scoring error:", err);
      setErrorMessage(
        err?.message || "Failed to score transaction. If the service is waking up, retry in a moment."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const applyPreset = (type: "normal" | "suspicious" | "transfer") => {
    setErrorMessage(null);
    if (type === "normal") {
      setAmount(45.0);
      setProductCD("W");
      setCard1(13926);
      setPEmail("gmail.com");
      setREmail("gmail.com");
    } else if (type === "suspicious") {
      setAmount(1450.0);
      setProductCD("C");
      setCard1(9500);
      setPEmail("anonymous.com");
      setREmail("protonmail.com");
    } else {
      setAmount(820.0);
      setProductCD("R");
      setCard1(7919);
      setPEmail("yahoo.com");
      setREmail("anonymous.com");
    }
  };

  return (
    <div className="card">
      {/* Toggle header */}
      <button
        type="button"
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
        }}
      >
        <span style={{ fontSize: "0.875rem", fontWeight: 700 }}>
          Try your own transaction ▼
        </span>
        <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
          {open ? "collapse" : "change values & score live model"}
        </span>
      </button>

      {open && (
        <div style={{ marginTop: "16px" }} className="fade-in">
          {/* Presets */}
          <div style={{ display: "flex", gap: "8px", marginBottom: "14px", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", alignSelf: "center" }}>
              Quick presets:
            </span>
            <button
              type="button"
              className="btn"
              style={{ fontSize: "0.75rem", padding: "4px 10px" }}
              onClick={() => applyPreset("normal")}
            >
              Normal transaction
            </button>
            <button
              type="button"
              className="btn"
              style={{
                fontSize: "0.75rem",
                padding: "4px 10px",
                color: "var(--risk-high)",
                borderColor: "var(--risk-high-border)",
              }}
              onClick={() => applyPreset("suspicious")}
            >
              Suspicious transaction
            </button>
            <button
              type="button"
              className="btn"
              style={{
                fontSize: "0.75rem",
                padding: "4px 10px",
                color: "var(--risk-med)",
                borderColor: "var(--risk-med-border)",
              }}
              onClick={() => applyPreset("transfer")}
            >
              Self-transfer (risky)
            </button>
          </div>

          {errorMessage && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 12px",
                background: "var(--risk-high-bg)",
                border: "1px solid var(--risk-high-border)",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.8rem",
                color: "var(--risk-high)",
                marginBottom: "12px",
              }}
            >
              <AlertCircle size={14} />
              <span>{errorMessage}</span>
            </div>
          )}

          <form onSubmit={handleScore} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            {/* Amount */}
            <div>
              <label style={{ display: "block", fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
                Amount ($)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                value={amount}
                onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                className="input"
                style={{ fontFamily: "var(--font-mono)" }}
                required
              />
            </div>

            {/* Product Channel */}
            <div>
              <label style={{ display: "block", fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
                Transaction type
              </label>
              <select
                value={productCD}
                onChange={(e) => setProductCD(e.target.value)}
                className="input"
              >
                <option value="W">W — Standard web retail</option>
                <option value="C">C — Cross-border payment</option>
                <option value="R">R — Recurring payment</option>
                <option value="H">H — High-value service</option>
                <option value="S">S — Subscription</option>
              </select>
            </div>

            {/* Card ID */}
            <div>
              <label style={{ display: "block", fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
                Card ID proxy (card1)
              </label>
              <input
                type="number"
                value={card1}
                onChange={(e) => setCard1(parseInt(e.target.value) || 1000)}
                className="input"
                style={{ fontFamily: "var(--font-mono)" }}
              />
            </div>

            {/* Purchaser email */}
            <div>
              <label style={{ display: "block", fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
                Purchaser email domain
              </label>
              <select value={pEmail} onChange={(e) => setPEmail(e.target.value)} className="input">
                <option value="gmail.com">gmail.com</option>
                <option value="yahoo.com">yahoo.com</option>
                <option value="hotmail.com">hotmail.com</option>
                <option value="anonymous.com">anonymous.com (high risk)</option>
                <option value="protonmail.com">protonmail.com (high risk)</option>
              </select>
            </div>

            <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "flex-end", marginTop: "4px" }}>
              <button
                type="submit"
                disabled={isLoading}
                className="btn btn-primary"
                style={{ padding: "9px 20px", fontSize: "0.875rem" }}
              >
                {isLoading ? (
                  <>
                    <RefreshCw size={14} className="spin" />
                    <span>Waking up the model…</span>
                  </>
                ) : (
                  <>
                    <Zap size={15} />
                    <span>Score live model</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
