"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldAlert, BarChart3, Terminal, Activity } from "lucide-react";

export function Navigation() {
  const pathname = usePathname();

  return (
    <header style={{
      borderBottom: "1px solid var(--border-subtle)",
      background: "rgba(7, 10, 19, 0.85)",
      backdropFilter: "blur(12px)",
      position: "sticky",
      top: 0,
      zIndex: 50,
    }}>
      <div className="container" style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        height: "72px",
      }}>
        {/* Logo & Identity */}
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div style={{
            width: "40px",
            height: "40px",
            borderRadius: "10px",
            background: "linear-gradient(135deg, #38bdf8, #3b82f6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 16px rgba(56, 189, 248, 0.3)",
          }}>
            <ShieldAlert size={22} color="#070a13" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontWeight: 700, fontSize: "1.1rem", letterSpacing: "-0.02em" }}>
                Fraud Risk Analytics
              </span>
              <span className="badge badge-cyan" style={{ fontSize: "0.65rem", padding: "2px 8px" }}>
                v1.0 Deployed
              </span>
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: 0 }}>
              Defensible Fintech Risk Scoring & Grounded Explanations
            </p>
          </div>
        </div>

        {/* 2-Page Navigation Links */}
        <nav style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Link
            href="/"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 16px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.9rem",
              fontWeight: 500,
              color: pathname === "/" ? "var(--accent-cyan)" : "var(--text-secondary)",
              background: pathname === "/" ? "var(--accent-cyan-bg)" : "transparent",
              border: pathname === "/" ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
              transition: "all var(--transition-fast)",
            }}
          >
            <Activity size={16} />
            <span>Interactive Demo</span>
          </Link>

          <Link
            href="/methodology"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 16px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.9rem",
              fontWeight: 500,
              color: pathname === "/methodology" ? "var(--accent-cyan)" : "var(--text-secondary)",
              background: pathname === "/methodology" ? "var(--accent-cyan-bg)" : "transparent",
              border: pathname === "/methodology" ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
              transition: "all var(--transition-fast)",
            }}
          >
            <BarChart3 size={16} />
            <span>Methodology & Analytics</span>
          </Link>
        </nav>
      </div>
    </header>
  );
}
