import "./globals.css";
import { Navigation } from "@/components/Navigation";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fraud Risk Analytics & Detection System",
  description: "Defensible fintech fraud risk analytics, cost-sensitive threshold optimization, TreeSHAP reason codes, and grounded explanations.",
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
        <footer style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "24px 0",
          marginTop: "60px",
          color: "var(--text-muted)",
          fontSize: "0.8rem",
          textAlign: "center",
        }}>
          <div className="container">
            <p>
              <strong>Fraud Risk Analytics & Detection System</strong> — Built entirely on free tiers of hosted services ($0 infrastructure stack).
            </p>
            <p style={{ marginTop: "4px" }}>
              IEEE-CIS Fraud Detection Dataset • Champion LightGBM • TreeSHAP Explainability • 100% Grounded Local GenAI Narratives
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
