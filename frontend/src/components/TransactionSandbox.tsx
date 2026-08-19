"use client";

import { useState } from "react";
import { scoreTransaction } from "@/lib/api";
import { Play, RotateCcw, Zap } from "lucide-react";

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

  const handleScore = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
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
      onScoreResult(result);
    } catch (err) {
      console.error("Sandbox scoring error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePreset = (type: "low" | "high" | "transfer") => {
    if (type === "low") {
      setAmount(45.0);
      setProductCD("W");
      setCard1(13926);
      setPEmail("gmail.com");
      setREmail("gmail.com");
    } else if (type === "high") {
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
    <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Interactive Scoring Sandbox</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Inject custom parameters to test real-time inference against the Champion model
          </p>
        </div>
        <div style={{ display: "flex", gap: "6px" }}>
          <button
            type="button"
            onClick={() => handlePreset("low")}
            style={{
              padding: "4px 8px",
              background: "var(--risk-low-bg)",
              border: "1px solid var(--risk-low-border)",
              color: "var(--risk-low)",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.75rem",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Low Risk Preset
          </button>
          <button
            type="button"
            onClick={() => handlePreset("high")}
            style={{
              padding: "4px 8px",
              background: "var(--risk-high-bg)",
              border: "1px solid var(--risk-high-border)",
              color: "var(--risk-high)",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.75rem",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            High Risk Preset
          </button>
        </div>
      </div>

      <form onSubmit={handleScore} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        {/* Transaction Amount */}
        <div>
          <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
            Transaction Amount ($ USD)
          </label>
          <input
            type="number"
            step="0.01"
            min="0.01"
            value={amount}
            onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-primary)",
              fontFamily: "var(--font-mono)",
            }}
            required
          />
        </div>

        {/* Product Channel */}
        <div>
          <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
            Product Channel (ProductCD)
          </label>
          <select
            value={productCD}
            onChange={(e) => setProductCD(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-primary)",
            }}
          >
            <option value="W">W — Standard Web Retail</option>
            <option value="C">C — High-Risk Cross-Border</option>
            <option value="R">R — Recurring Merchant</option>
            <option value="H">H — High-Value Service</option>
            <option value="S">S — Subscription Channel</option>
          </select>
        </div>

        {/* Card 1 Proxy */}
        <div>
          <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
            Card Entity Proxy (card1)
          </label>
          <input
            type="number"
            value={card1}
            onChange={(e) => setCard1(parseInt(e.target.value) || 1000)}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-primary)",
              fontFamily: "var(--font-mono)",
            }}
          />
        </div>

        {/* Email Domains */}
        <div>
          <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
            Purchaser Email Domain
          </label>
          <select
            value={pEmail}
            onChange={(e) => setPEmail(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-primary)",
            }}
          >
            <option value="gmail.com">gmail.com (Standard)</option>
            <option value="yahoo.com">yahoo.com (Standard)</option>
            <option value="anonymous.com">anonymous.com (High Risk)</option>
            <option value="protonmail.com">protonmail.com (High Risk)</option>
            <option value="hotmail.com">hotmail.com</option>
          </select>
        </div>

        <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "flex-end", marginTop: "4px" }}>
          <button
            type="submit"
            disabled={isLoading}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "10px 20px",
              background: "linear-gradient(135deg, #38bdf8, #3b82f6)",
              color: "#070a13",
              fontWeight: 700,
              fontSize: "0.9rem",
              border: "none",
              borderRadius: "var(--radius-md)",
              cursor: isLoading ? "not-allowed" : "pointer",
              opacity: isLoading ? 0.7 : 1,
              boxShadow: "0 0 16px rgba(56, 189, 248, 0.3)",
              transition: "transform var(--transition-fast)",
            }}
          >
            <Zap size={16} />
            <span>{isLoading ? "Running Inference..." : "Score Transaction"}</span>
          </button>
        </div>
      </form>
    </div>
  );
}
