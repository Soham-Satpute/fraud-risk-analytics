import "./globals.css";
import { Navigation } from "@/components/Navigation";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fraud Risk Analytics & Detection System",
  description: "End-to-end fraud risk analytics and modeling with LightGBM, SHAP explainability, and defensible business decision analysis — built on a $0 free-tier stack.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Navigation />
        <main>{children}</main>
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "24px 0",
            marginTop: "60px",
            color: "var(--text-muted)",
            fontSize: "0.78rem",
            textAlign: "center",
          }}
        >
          <div className="container">
            <p>
              <strong style={{ color: "var(--text-secondary)" }}>Fraud Risk Analytics &amp; Detection System</strong>
              {" "}· Portfolio-scale deployed demo · Built entirely on free-tier infrastructure
            </p>
            <p style={{ marginTop: "4px" }}>
              IEEE-CIS Dataset · LightGBM · TreeSHAP Explainability · Offline-Generated Analyst Narratives
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
