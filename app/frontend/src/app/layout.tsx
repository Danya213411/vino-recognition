import type { Metadata, Viewport } from "next";
import { SiteHeader } from "@/shared/ui/SiteHeader";
import { SiteFooter } from "@/shared/ui/SiteFooter";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "ХАКАТОН — распознавание", template: "%s — ХАКАТОН" },
  description: "Распознавание российского вина по фотографии этикетки.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#fefdfa",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ru">
      <body>
        <div className="page-shell">
          <SiteHeader />
          {children}
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
