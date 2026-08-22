"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck, ExternalLink } from "lucide-react";

export function Navigation() {
  const pathname = usePathname();

  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        background: "var(--bg-app)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      <div
        className="container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: "60px",
        }}
      >
        {/* Logo */}
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: "10px", textDecoration: "none" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "var(--radius-md)",
              background: "var(--accent-blue)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ShieldCheck size={18} color="#fff" strokeWidth={2.5} />
          </div>
          <div>
            <span style={{ fontWeight: 700, fontSize: "0.95rem", color: "var(--text-primary)" }}>
              Fraud Risk Analytics
            </span>
            <span
              style={{
                display: "block",
                fontSize: "0.7rem",
                color: "var(--text-muted)",
                lineHeight: 1.2,
              }}
            >
              Portfolio-scale deployed demo
            </span>
          </div>
        </Link>

        {/* Nav links */}
        <nav style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <Link
            href="/"
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.875rem",
              fontWeight: 500,
              color: pathname === "/" ? "var(--text-primary)" : "var(--text-secondary)",
              borderBottom: pathname === "/" ? "2px solid var(--accent-blue)" : "2px solid transparent",
              transition: "color var(--transition)",
            }}
          >
            Story
          </Link>
          <Link
            href="/demo"
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.875rem",
              fontWeight: 500,
              color: pathname.startsWith("/demo") ? "var(--text-primary)" : "var(--text-secondary)",
              borderBottom: pathname.startsWith("/demo") ? "2px solid var(--accent-blue)" : "2px solid transparent",
              transition: "color var(--transition)",
            }}
          >
            Live Demo
          </Link>
          <a
            href="https://github.com/Soham-Satpute/fraud-risk-analytics/blob/main/case-study/fraud-risk-case-study.md"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.875rem",
              fontWeight: 500,
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "4px",
              transition: "color var(--transition)",
            }}
          >
            <span>Case Study</span>
            <ExternalLink size={12} opacity={0.7} />
          </a>
          <a
            href="https://github.com/Soham-Satpute/fraud-risk-analytics"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-md)",
              fontSize: "0.875rem",
              fontWeight: 500,
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "4px",
              transition: "color var(--transition)",
            }}
          >
            <span>GitHub</span>
            <ExternalLink size={12} opacity={0.7} />
          </a>
        </nav>
      </div>
    </header>
  );
}
