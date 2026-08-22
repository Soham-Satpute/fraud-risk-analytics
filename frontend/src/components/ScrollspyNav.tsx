"use client";

import React from "react";
import { useScrollspy } from "@/hooks/useScrollspy";

export interface NavSection {
  id: string;
  label: string;
}

interface ScrollspyNavProps {
  sections: NavSection[];
}

export function ScrollspyNav({ sections }: ScrollspyNavProps) {
  const sectionIds = sections.map((s) => s.id);
  const activeId = useScrollspy(sectionIds, 120);

  const scrollToSection = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault();
    const elem = document.getElementById(id);
    if (elem) {
      const topOffset = 80;
      const elementPosition = elem.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - topOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });
    }
  };

  return (
    <nav className="scrollspy-nav" aria-label="Table of contents">
      <div style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "12px" }}>
        Contents
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "6px" }}>
        {sections.map((section, index) => {
          const isActive = activeId === section.id;
          return (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                onClick={(e) => scrollToSection(e, section.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  fontSize: "0.82rem",
                  color: isActive ? "var(--accent-blue)" : "var(--text-secondary)",
                  fontWeight: isActive ? 600 : 400,
                  padding: "4px 8px",
                  borderRadius: "var(--radius-sm)",
                  background: isActive ? "var(--accent-blue-bg)" : "transparent",
                  transition: "all var(--transition)",
                  textDecoration: "none",
                }}
              >
                <span style={{ fontSize: "0.7rem", opacity: 0.6, fontFamily: "var(--font-mono)" }}>
                  0{index + 1}
                </span>
                <span>{section.label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
