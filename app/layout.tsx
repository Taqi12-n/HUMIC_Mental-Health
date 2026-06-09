import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MindVoice AI - Detect Mental Health Patterns Through Voice",
  description:
    "AI-powered mental health detection through advanced voice analysis. Research-grade audio analysis using machine learning and deep learning models.",
  keywords: ["mental health", "AI", "voice analysis", "audio detection", "machine learning"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="antialiased">
      <body className="min-h-screen flex flex-col">{children}</body>
    </html>
  );
}
