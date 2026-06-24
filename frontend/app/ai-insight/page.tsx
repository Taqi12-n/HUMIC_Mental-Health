"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { RefreshCw, Brain, Sparkles, ArrowRight, Home } from "lucide-react";
import Link from "next/link";
import XaiSection from "@/components/XaiSection";
import { getApiUrl } from "@/utils/api";

// Fallback mockup matching reference specifications
const fallbackData = {
  id: "fallback-mock-id",
  filename: "mental_health_sample.wav",
  date: "5/13/2026",
  timestamp: "5/13/2026, 11:02:50 PM",
  primaryDetection: "Depression",
  confidence: 78,
  metrics: {
    depression: 78,
    normal: 22
  },
  audioInfo: {
    duration: "45.0s",
    avgPitch: "152 Hz",
    energyLevel: "Medium",
    signalQuality: "94%"
  },
  performance: {
    accuracy: "92.4%",
    precision: "89.7%",
    f1Score: "90.8%"
  }
};

function AiInsightContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const resultId = searchParams.get("id");

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Determine active ID: query param first, then localStorage fallback
    const activeId = resultId || localStorage.getItem("mindvoice_active_result_id");

    if (!activeId) {
      router.push("/#upload");
      return;
    }

    if (activeId === "fallback-mock-id") {
      setData(fallbackData);
      setLoading(false);
      return;
    }

    const fetchResult = async () => {
      try {
        const response = await fetch(getApiUrl(`/api/results/${activeId}`));
        if (!response.ok) throw new Error("Result not found");
        const json = await response.json();
        setData(json);
      } catch (err) {
        console.warn("Backend fetch failed, using fallback mock data for preview.", err);
        setData(fallbackData);
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [resultId]);

  if (loading) {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center p-4">
        <RefreshCw className="animate-spin text-primary mb-4" size={32} />
        <p className="text-text-muted font-medium">Loading Explainable AI (XAI) models...</p>
      </div>
    );
  }

  // Empty State if no result is available
  if (!data) {
    return (
      <div className="max-w-md mx-auto text-center px-4 py-16 space-y-6">
        <div className="w-16 h-16 rounded-2xl gradient-bg-subtle flex items-center justify-center text-primary mx-auto shadow-sm">
          <Brain size={32} className="animate-pulse" />
        </div>
        <div className="space-y-2">
          <h3 className="text-xl font-extrabold text-text">No Active Analysis Found</h3>
          <p className="text-sm text-text-muted leading-relaxed">
            We couldn't find an active audio analysis session. Please upload your voice recording on the home page first to view AI insights.
          </p>
        </div>
        <div className="pt-4">
          <button
            onClick={() => router.push("/#upload")}
            className="w-full inline-flex items-center justify-center gap-2 py-3.5 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform cursor-pointer"
          >
            <Sparkles size={16} />
            Start Voice Analysis
            <ArrowRight size={16} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10 space-y-6 sm:space-y-8">
      {/* Page Title & Breadcrumbs */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-text-muted mb-2">
          <span>AI Insight Panel</span>
          <span>•</span>
          <span>SHAP & LIME Interpretability</span>
          {data.filename && (
            <>
              <span>•</span>
              <span className="text-primary truncate max-w-[150px]">{data.filename}</span>
            </>
          )}
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-text mb-1">Explainable AI Insights</h2>
        <p className="text-text-muted text-sm sm:text-base">
          Detailed breakdown of acoustic biomarker contributions driving the model's prediction
        </p>
      </div>

      {/* Main XAI Component */}
      <XaiSection data={data} />

      {/* Bottom Nav Links */}
      <div className="flex flex-col sm:flex-row gap-3 pt-4 justify-between items-center border-t border-border-light">
        <p className="text-xs text-text-muted leading-relaxed text-center sm:text-left">
          These interpretations are generated locally. Read the descriptions inside the Biomarker Dictionary to understand the clinical significance of each factor.
        </p>
        <div className="flex gap-3 w-full sm:w-auto shrink-0 mt-3 sm:mt-0">
          <Link
            href={resultId ? `/results?id=${resultId}` : "/results"}
            className="flex-1 sm:flex-initial text-center px-6 py-3 rounded-full border border-border bg-white text-text font-bold text-sm hover:bg-bg transition-colors"
          >
            Back to Results
          </Link>
          <Link
            href="/"
            className="flex-1 sm:flex-initial text-center px-6 py-3 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
          >
            Go to Home
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function AiInsightPage() {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const id = localStorage.getItem("mindvoice_active_result_id");
    setActiveId(id);
  }, []);

  return (
    <>
      {/* Sticky header matching landing navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-navbar shadow-sm border-b border-white/50 h-16 sm:h-20 flex items-center">
        <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full gradient-bg flex items-center justify-center shadow-md">
              <div className="w-4 h-4 bg-white/30 rounded-full" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold text-text leading-tight">MindVoice AI</h1>
              <p className="text-[10px] sm:text-xs text-text-muted leading-tight">Telkom University</p>
            </div>
          </Link>

          <div className="flex items-center gap-1">
            <Link href="/" className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-text-muted hover:text-text">
              Home
            </Link>
            <Link 
              href={activeId ? `/results?id=${activeId}` : "/results"} 
              className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-text-muted hover:text-text"
            >
              Results
            </Link>
            <span className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-white gradient-bg shadow-sm">
              AI Insights
            </span>
          </div>
        </div>
      </nav>

      {/* Main Container */}
      <main className="min-h-screen pt-20 sm:pt-24 pb-16 bg-bg">
        <Suspense fallback={
          <div className="min-h-screen bg-bg flex flex-col items-center justify-center p-4">
            <RefreshCw className="animate-spin text-primary mb-4" size={32} />
            <p className="text-text-muted font-medium">Loading page modules...</p>
          </div>
        }>
          <AiInsightContent />
        </Suspense>
      </main>
    </>
  );
}
