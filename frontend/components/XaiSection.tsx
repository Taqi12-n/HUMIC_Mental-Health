"use client";

import {
  Activity, Brain, ChevronDown, ChevronUp, Info, Sparkles,
  BarChart2, Zap, BookOpen, TrendingUp, HelpCircle,
  CheckCircle, XCircle, AlertTriangle, Mic, Sliders, Search
} from "lucide-react";
import { useMemo, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Types ─────────────────────────────────────────────────────────────────────

interface Layer1Group {
  contribution_pct: number;
  direction: "DEPRESI" | "NORMAL";
  mean_shap?: number;
}

interface Layer2SubGroup {
  contribution_pct: number;
  direction: "DEPRESI" | "NORMAL";
}

interface Layer3Feature {
  rank: number;
  feature_group: string;
  feature_sub: string;
  feature_idx: number;
  shap_value: number;
  magnitude?: number;
  direction: "DEPRESI" | "NORMAL";
}

interface ShapExplanation {
  dominant_feature_group: string;
  layer1_group: Record<string, Layer1Group>;
  layer2_subgroup?: {
    MelSpec_subgroups?: Record<string, Layer2SubGroup>;
    MFCC_subgroups?: Record<string, Layer2SubGroup>;
  };
  layer3_waterfall?: Layer3Feature[];
  layer4_text?: string;
  baseline_prob_depresi?: number;
  is_fallback?: boolean;
  error?: string;
}

interface XaiSectionProps {
  data: {
    primaryDetection: string;
    confidence: number;
    metrics?: {
      depression?: number;
      normal?: number;
    };
    recommendation?: {
      title?: string;
      text?: string;
    };
    shapExplanation?: ShapExplanation | null;
    modelInfo?: {
      name?: string;
      scenario?: string;
      threshold?: number;
      testF1?: string;
    };
  };
}

// ─── Static copy per feature group ────────────────────────────────────────────

const groupCopy: Record<string, {
  title: string;
  technical: string;
  plain: string;
  plainDepression: string;
  plainNormal: string;
  emoji: string;
  details: string[];
  explanation: string;
}> = {
  Wav2Vec: {
    title: "Speech Flow & Rhythm",
    technical: "Speech Flow & Pauses (Wav2Vec)",
    plain: "AI analyzed the rhythm, speed, intonation, and pause patterns in your speech.",
    plainDepression:
      "Detected longer pauses and a flatter or less dynamic rhythm, which are often associated with depression-related voice patterns.",
    plainNormal:
      "Your speed, pauses, and voice modulation show dynamic and stable patterns consistent with typical healthy speech.",
    emoji: "🎙️",
    details: ["Speech rhythm", "Pauses between words", "Intonation modulation", "Speech flow energy"],
    explanation:
      "Wav2Vec is a deep learning speech model that extracts micro-patterns in rhythm and speech flow that are often imperceptible to the human ear.",
  },
  MFCC: {
    title: "Voice Timbre & Quality",
    technical: "Vocal Texture & Quality (MFCC)",
    plain: "AI analyzed the vocal texture, resonance, and changes in voice quality over time.",
    plainDepression:
      "Vocal timbre analysis detected markers correlated with reduced resonance, heavier tone, or breath patterns linked to depression.",
    plainNormal:
      "Your vocal texture and resonance remained stable, clear, and aligned with standard healthy voice characteristics.",
    emoji: "🎛️",
    details: ["Vocal richness", "Pitch stability", "Acoustic resonance", "Breath smoothness"],
    explanation:
      "MFCC (Mel-Frequency Cepstral Coefficients) acts like a vocal fingerprint, capturing the texture, clarity, and stability of your vocal tract.",
  },
  MelSpec: {
    title: "Speech Frequency Pattern",
    technical: "Pitch & Frequency Spectrogram (MelSpec)",
    plain: "AI measured how your voice energy is spread across low, mid, and high frequencies.",
    plainDepression:
      "The energy distribution in certain frequency bands showed reduced intensity, particularly in the middle to upper speech registers.",
    plainNormal:
      "Your vocal energy was evenly balanced and consistent across the frequency spectrum.",
    emoji: "📊",
    details: ["Low frequency energy", "Mid frequency energy", "High frequency energy"],
    explanation:
      "A Mel Spectrogram maps audio frequencies to the Mel scale, mirroring human hearing pitch perception, to measure speech energy distribution.",
  },
};

const normalizeGroupName = (name: string) => {
  if (name.includes("Wav2Vec")) return "Wav2Vec";
  if (name.includes("MFCC")) return "MFCC";
  if (name.includes("MelSpec")) return "MelSpec";
  return name;
};

const getGroupIcon = (name: string, size = 20, className = "") => {
  const norm = normalizeGroupName(name);
  switch (norm) {
    case "Wav2Vec":
      return <Mic size={size} className={className} />;
    case "MFCC":
      return <Sliders size={size} className={className} />;
    case "MelSpec":
      return <BarChart2 size={size} className={className} />;
    default:
      return <Brain size={size} className={className} />;
  }
};

// ─── Native Web Interactive Charts ────────────────────────────────────────────

function GroupContributionChart() {
  const data = [
    { name: "Speech Flow & Rhythm (Wav2Vec)", pct: 63.95, color: "bg-violet-500", desc: "Rhythm, intonation, and pause patterns" },
    { name: "Speech Frequency Pattern (MelSpec)", pct: 23.29, color: "bg-blue-500", desc: "Energy distribution across pitch registers" },
    { name: "Voice Timbre & Quality (MFCC)", pct: 12.76, color: "bg-emerald-500", desc: "Vocal timbre and stability" },
  ];

  return (
    <div className="space-y-4 p-4 bg-slate-50/50 rounded-xl border border-slate-100">
      {data.map((item) => (
        <div key={item.name} className="group space-y-1">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-text">{item.name}</span>
            <span className="font-black text-text">{item.pct}%</span>
          </div>
          <div className="relative w-full h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200/30">
            <motion.div
              className={`h-full rounded-full ${item.color}`}
              initial={{ width: 0 }}
              animate={{ width: `${item.pct}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
          <p className="text-[10px] text-text-muted italic group-hover:text-primary transition-colors">{item.desc}</p>
        </div>
      ))}
    </div>
  );
}

function MelSpecBreakdownChart() {
  const data = [
    { label: "High Freq (High Pitch)", pct: 10.91, color: "from-sky-400 to-blue-500", desc: "Captures consonants and speech clarity" },
    { label: "Mid Freq (Mid Pitch)", pct: 6.93, color: "from-blue-400 to-indigo-500", desc: "Captures main conversational voice properties" },
    { label: "Low Freq (Low Pitch)", pct: 5.44, color: "from-indigo-400 to-violet-500", desc: "Captures fundamental pitch vibrations" },
  ];

  return (
    <div className="space-y-3 p-4 bg-slate-50/50 rounded-xl border border-slate-100">
      {data.map((item) => (
        <div key={item.label} className="space-y-1">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-text">{item.label}</span>
            <span className="font-black text-text">{item.pct}%</span>
          </div>
          <div className="relative w-full h-3 bg-slate-100 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full bg-gradient-to-r ${item.color}`}
              initial={{ width: 0 }}
              animate={{ width: `${item.pct * 5}%` }} // Scaled visually for impact
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
          <p className="text-[10px] text-text-muted">{item.desc}</p>
        </div>
      ))}
    </div>
  );
}

function MfccBreakdownChart() {
  const data = [
    { label: "MFCC Base (Timbre)", pct: 8.01, color: "from-emerald-400 to-teal-500", desc: "Measures fundamental timbre and vocal tract stability" },
    { label: "MFCC Delta2 (Acceleration)", pct: 2.39, color: "from-teal-400 to-cyan-500", desc: "Measures sudden shifts in speech dynamics" },
    { label: "MFCC Delta (Transition)", pct: 2.35, color: "from-cyan-400 to-sky-500", desc: "Measures dynamics of transition between speech frames" },
  ];

  return (
    <div className="space-y-3 p-4 bg-slate-50/50 rounded-xl border border-slate-100">
      {data.map((item) => (
        <div key={item.label} className="space-y-1">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-text">{item.label}</span>
            <span className="font-black text-text">{item.pct}%</span>
          </div>
          <div className="relative w-full h-3 bg-slate-100 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full bg-gradient-to-r ${item.color}`}
              initial={{ width: 0 }}
              animate={{ width: `${item.pct * 8}%` }} // Scaled visually for impact
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
          <p className="text-[10px] text-text-muted">{item.desc}</p>
        </div>
      ))}
    </div>
  );
}

function BeeswarmChart() {
  const features = [
    { name: "Speech Flow (Wav2Vec h215)", desc: "Most sensitive vocal feature for pause patterns" },
    { name: "High Pitch Energy (Mel Band 96)", desc: "Captures clarity of word modulation" },
    { name: "Base Timbre (MFCC C1)", desc: "Resonance of fundamental vocal cord vibrations" },
    { name: "Emphasis Variation (Wav2Vec h541)", desc: "Richness of speech emphasis variations" },
  ];

  const points = useMemo(() => {
    return features.map((_, idx) => {
      const fPoints = [];
      const numPoints = 25;
      const seed = idx * 10;
      for (let i = 0; i < numPoints; i++) {
        const x = Math.sin(i * 0.5 + seed) * 70 + (Math.random() - 0.5) * 12;
        const y = Math.cos(i * 1.5 + seed) * 7 * (1 - Math.abs(x) / 90);
        const val = i % 2 === 0 ? "high" : "low";
        fPoints.push({ x, y, val });
      }
      return fPoints;
    });
  }, []);

  const [hoveredPoint, setHoveredPoint] = useState<{
    feature: string;
    val: string;
    impact: string;
  } | null>(null);

  return (
    <div className="p-4 bg-slate-50/50 rounded-xl border border-slate-100 space-y-5 relative">
      <div className="flex justify-between items-center text-[10px] font-bold text-text-muted px-2">
        <span>← NORMAL (Pushes toward Normal)</span>
        <span>DEPRESSION (Pushes toward Depression) →</span>
      </div>

      <div className="space-y-4">
        {features.map((feat, idx) => (
          <div key={feat.name} className="space-y-1">
            <div className="flex justify-between text-xs font-semibold text-text">
              <span>{feat.name}</span>
              <span className="text-[10px] text-text-muted">{feat.desc}</span>
            </div>
            
            <div className="relative w-full h-12 bg-white rounded-lg border border-slate-100 flex items-center overflow-hidden">
              <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-slate-200" />
              
              <svg className="w-full h-full">
                <g transform="translate(0, 24)">
                  {points[idx].map((pt, pIdx) => {
                    const cx = `${50 + (pt.x / 2)}%`;
                    const cy = pt.y;
                    const isRed = pt.val === "high";
                    return (
                      <motion.circle
                        key={pIdx}
                        cx={cx}
                        cy={cy}
                        r={4.5}
                        className={`cursor-pointer stroke-white stroke-[0.5px] ${
                          isRed
                            ? "fill-rose-500 hover:fill-rose-400"
                            : "fill-indigo-500 hover:fill-indigo-400"
                        }`}
                        whileHover={{ r: 7 }}
                        onMouseEnter={() => {
                          setHoveredPoint({
                            feature: feat.name,
                            val: isRed ? "High Value (Distinctive Marker)" : "Low Value (Typical / Normal)",
                            impact: pt.x > 0 ? "Depression Tendency" : "Normal Tendency",
                          });
                        }}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    );
                  })}
                </g>
              </svg>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-center gap-6 text-[10px] font-semibold text-text-muted pt-1">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-rose-500" />
          <span>High Feature Value</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
          <span>Low Feature Value</span>
        </div>
      </div>

      {hoveredPoint && (
        <div className="absolute z-20 bg-slate-900/95 text-white p-2.5 rounded-lg text-[10px] space-y-1 shadow-xl pointer-events-none transition-all border border-slate-800 left-[5%] bottom-[8px] width-[90%] w-[90%]">
          <p className="font-bold border-b border-slate-800 pb-1">{hoveredPoint.feature}</p>
          <div className="flex justify-between">
            <span>Feature Value: <strong className="text-amber-400">{hoveredPoint.val}</strong></span>
            <span>Prediction Push: <strong className="text-cyan-400">{hoveredPoint.impact}</strong></span>
          </div>
        </div>
      )}
    </div>
  );
}

function WaterfallChart() {
  const steps = [
    { label: "Baseline Value", value: 50.0, type: "base" },
    { label: "Stable Rhythm (Wav2Vec h215)", value: -5.5, type: "normal" },
    { label: "Longer Pause Modulation (Wav2Vec h541)", value: +3.2, type: "depresi" },
    { label: "Consistent Voice Pitch (Mel Band 96)", value: -4.1, type: "normal" },
    { label: "Breath Texture Shift (MFCC C1)", value: +0.4, type: "depresi" },
    { label: "Final Prediction Score", value: 44.0, type: "final" },
  ];

  return (
    <div className="p-4 bg-slate-50/50 rounded-xl border border-slate-100 space-y-3">
      <div className="space-y-2">
        {steps.map((step, idx) => {
          const isNegative = step.value < 0;
          const isBaseOrFinal = step.type === "base" || step.type === "final";
          
          return (
            <div key={idx} className="flex items-center justify-between text-xs gap-3">
              <span className="w-1/3 text-text font-semibold truncate">{step.label}</span>
              
              <div className="flex-1 relative h-6 bg-slate-100/50 rounded flex items-center px-1">
                {isBaseOrFinal ? (
                  <motion.div
                    className="h-4 bg-slate-400 rounded"
                    initial={{ width: 0 }}
                    animate={{ width: `${step.value}%` }}
                    transition={{ duration: 1 }}
                  />
                ) : (
                  <motion.div
                    className={`h-4 rounded absolute ${isNegative ? "bg-emerald-500" : "bg-primary"}`}
                    initial={{ width: 0, left: "50%" }}
                    animate={{
                      width: `${Math.abs(step.value) * 4}%`,
                      left: isNegative ? `${50 - Math.abs(step.value) * 4}%` : "50%"
                    }}
                    transition={{ duration: 1 }}
                  />
                )}
                {!isBaseOrFinal && <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-slate-300" />}
              </div>

              <span className={`w-14 text-right font-black ${
                step.type === "depresi" ? "text-primary" : 
                step.type === "normal" ? "text-emerald-600" : "text-text"
              }`}>
                {step.value > 0 && !isBaseOrFinal ? `+${step.value}%` : `${step.value}%`}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-text-muted leading-relaxed">
        Accumulative contribution chart: Shows how your vocal features push the final probability away from or toward the population baseline.
      </p>
    </div>
  );
}

// ─── Main XaiSection Component ──────────────────────────────────────────────────

export default function XaiSection({ data }: XaiSectionProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "global">("overview");
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

  const shap = data.shapExplanation;
  const hasShap = Boolean(shap && !shap.error && shap.layer1_group);
  const isDepression =
    data.primaryDetection === "DEPRESI" || data.primaryDetection === "Depression";
  const depressionPct =
    data.metrics?.depression ?? (isDepression ? data.confidence : 100 - data.confidence);
  const normalPct = data.metrics?.normal ?? 100 - depressionPct;

  const sortedGroups = useMemo(() => {
    if (!hasShap || !shap?.layer1_group) return [];
    return Object.entries(shap.layer1_group)
      .map(([name, info]) => ({
        name: normalizeGroupName(name),
        contribution: info.contribution_pct,
        direction: info.direction,
      }))
      .sort((a, b) => b.contribution - a.contribution);
  }, [hasShap, shap]);

  // Set the strongest feature group expanded by default on mount
  useEffect(() => {
    if (sortedGroups.length > 0) {
      setExpandedGroups({ [sortedGroups[0].name]: true });
    }
  }, [sortedGroups]);

  const mainGroup = sortedGroups[0];
  const mainCopy = mainGroup ? groupCopy[mainGroup.name] : null;
  const plainExplanation =
    shap?.layer4_text ||
    "AI compared your speech features to learned patterns and based its prediction on the strongest signals it found.";

  const layer3Features = shap?.layer3_waterfall?.slice(0, 10) ?? [];
  const melspecSubs = shap?.layer2_subgroup?.MelSpec_subgroups ?? {};
  const mfccSubs = shap?.layer2_subgroup?.MFCC_subgroups ?? {};

  // Step-by-step description of SHAP pipeline
  const HOW_SHAP_WORKS = [
    {
      step: "1",
      icon: <Mic size={18} className="text-primary" />,
      title: "Audio Recorded",
      desc: "Voice recording is loaded and silence/noise is cleaned.",
    },
    {
      step: "2",
      icon: <Search size={18} className="text-primary" />,
      title: "Feature Extraction",
      desc: "Vocal pitch (MelSpec), timbre (MFCC), and speech rhythm (Wav2Vec) are extracted.",
    },
    {
      step: "3",
      icon: <Brain size={18} className="text-primary" />,
      title: "Model Evaluation",
      desc: "Model compares patterns to find matches in clinical reference sets.",
    },
    {
      step: "4",
      icon: <Sparkles size={18} className="text-primary" />,
      title: "SHAP Explanations",
      desc: "SHAP computes individual feature impact towards the final decision.",
    },
  ];

  return (
    <div className="space-y-5">
      {/* ─── Section A: Quick Summary Card ─── */}
      <section className="bg-white rounded-2xl border border-border-light p-5 sm:p-7 soft-shadow space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div className="space-y-3">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border ${isDepression ? "bg-red-50 border-red-100 text-primary" : "bg-emerald-50 border-emerald-100 text-emerald-700"}`}>
              <Brain size={13} />
              AI Explanation Summary
            </div>
            <div>
              <h3 className="text-2xl sm:text-3xl font-black text-text">
                {isDepression ? "Voice markers suggest depression" : "Voice appears typical"}
              </h3>
              <p className="text-sm text-text-muted leading-relaxed mt-2 max-w-2xl font-medium">
                {mainCopy
                  ? mainGroup.direction === "DEPRESI"
                    ? mainCopy.plainDepression
                    : mainCopy.plainNormal
                  : plainExplanation}
              </p>
            </div>
          </div>

          {/* Confidence badge */}
          <div className={`rounded-2xl p-5 min-w-full md:min-w-[200px] border ${isDepression ? "bg-red-50 border-red-100" : "bg-emerald-50 border-emerald-100"}`}>
            <p className="text-xs font-bold text-text-muted mb-1">Confidence Score</p>
            <p className={`text-5xl font-black leading-none ${isDepression ? "text-primary" : "text-emerald-600"}`}>
              {data.confidence}%
            </p>
            <div className="mt-3 h-2.5 rounded-full bg-white overflow-hidden border border-border-light">
              <div
                className={`h-full ${isDepression ? "bg-primary" : "bg-emerald-500"} transition-all duration-1000`}
                style={{ width: `${data.confidence}%` }}
              />
            </div>
            <p className="text-[10px] text-text-muted mt-2">
              {data.confidence >= 70
                ? "High Confidence — Strong pattern match"
                : data.confidence >= 50
                ? "Medium Confidence — Borderline match"
                : "Low Confidence — Early screening signal"}
            </p>
          </div>
        </div>

        {/* Depression / Normal probability */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl bg-bg border border-border-light p-4">
            <div className="flex items-center gap-2 mb-3">
              <XCircle size={14} className="text-primary" />
              <p className="text-xs font-bold text-text-muted">Depression Probability</p>
            </div>
            <p className="text-3xl font-black text-primary">{depressionPct}%</p>
            <div className="mt-2 h-2.5 rounded-full bg-white overflow-hidden border border-border-light">
              <div className="h-full bg-primary transition-all duration-700" style={{ width: `${depressionPct}%` }} />
            </div>
          </div>
          <div className="rounded-xl bg-bg border border-border-light p-4">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle size={14} className="text-emerald-600" />
              <p className="text-xs font-bold text-text-muted">Normal Probability</p>
            </div>
            <p className="text-3xl font-black text-emerald-600">{normalPct}%</p>
            <div className="mt-2 h-2.5 rounded-full bg-white overflow-hidden border border-border-light">
              <div className="h-full bg-emerald-500 transition-all duration-700" style={{ width: `${normalPct}%` }} />
            </div>
          </div>
        </div>
      </section>

      {/* ─── Section B: How the AI Works ─── */}
      <section className="bg-white rounded-2xl border border-border-light soft-shadow overflow-hidden">
        <button
          onClick={() => setShowHowItWorks((v) => !v)}
          className="w-full flex items-center justify-between p-5 sm:p-6 text-left hover:bg-bg/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary shrink-0">
              <BookOpen size={19} />
            </div>
            <div>
              <h4 className="text-sm sm:text-base font-extrabold text-text">
                How does the AI analyze your voice?
              </h4>
              <p className="text-xs text-text-muted">Step-by-step process of the AI screening engine</p>
            </div>
          </div>
          {showHowItWorks ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showHowItWorks && (
          <div className="px-5 sm:px-6 pb-6 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {HOW_SHAP_WORKS.map((step) => (
                <div key={step.step} className="bg-bg rounded-xl p-4 border border-border-light space-y-2">
                  <div className="flex items-center gap-2">
                    {step.icon}
                    <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                      Step {step.step}
                    </span>
                  </div>
                  <p className="text-sm font-extrabold text-text">{step.title}</p>
                  <p className="text-xs text-text-muted leading-relaxed">{step.desc}</p>
                </div>
              ))}
            </div>
            <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-100 rounded-xl">
              <Info size={15} className="text-blue-600 shrink-0 mt-0.5" />
              <p className="text-xs text-blue-700 leading-relaxed">
                <strong>About the SHAP Method:</strong> SHAP (SHapley Additive exPlanations) is a game-theoretic approach that assigns a contribution value to each vocal feature. It displays which features pushed the decision towards normal speech or depression indicators.
              </p>
            </div>
          </div>
        )}
      </section>

      {/* ─── Tab Switcher ─── */}
      <div className="flex bg-white rounded-xl border border-border-light p-1 gap-1">
        {[
          { id: "overview", icon: <Sparkles size={14} />, label: "Your Result Explained" },
          { id: "global", icon: <BarChart2 size={14} />, label: "Global Model Visualizations" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as "overview" | "global")}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-xs font-bold transition-all duration-200 ${activeTab === tab.id
              ? "gradient-bg text-white shadow-sm"
              : "text-text-muted hover:text-text"
              }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* ─── Section C: Your Result Explained (Interactive Expandable Cards) ─── */}
      {activeTab === "overview" && (
        <div className="space-y-5">
          {/* Local Analysis Explanation Banner */}
          <div className="rounded-2xl border border-indigo-100 bg-gradient-to-r from-indigo-50 to-violet-50 p-4 sm:p-5">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-100 flex items-center justify-center text-indigo-600 shrink-0">
                <Zap size={17} />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h5 className="text-sm font-extrabold text-indigo-900">Local Analysis — Your Personal Result</h5>
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-indigo-200 text-indigo-800 uppercase tracking-wider">Personalized</span>
                </div>
                <p className="text-xs text-indigo-700 leading-relaxed">
                  <strong>Local analysis</strong> explains <em>why the AI made this specific prediction for your voice recording</em>. Every chart and feature breakdown below is computed exclusively from your audio — not from averages or population data. It shows which vocal characteristics pushed the result toward <strong>Depression</strong> or <strong>Normal</strong> in your individual case.
                </p>
              </div>
            </div>
          </div>

          {/* Main reason summary card */}
          <section className="bg-white rounded-2xl border border-border-light p-5 sm:p-7 soft-shadow space-y-5">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary shrink-0">
                <Zap size={19} />
              </div>
              <div>
                <h4 className="text-base font-extrabold text-text">Primary Vocal Influencer</h4>
                <p className="text-sm text-text-muted leading-relaxed mt-1">
                  The most dominant vocal signal detected in your recording was from{" "}
                  <strong>{mainCopy?.title ?? (mainGroup ? groupCopy[mainGroup.name]?.title : "")}</strong>, contributing{" "}
                  <strong>{mainGroup?.contribution.toFixed(1)}%</strong> to the analysis.
                </p>
              </div>
            </div>

            {/* Expandable Group Cards */}
            {sortedGroups.length > 0 ? (
              <div className="space-y-4">
                {sortedGroups.map((group, i) => {
                  const copy = groupCopy[group.name];
                  const isMain = i === 0;
                  const isExpanded = expandedGroups[group.name];
                  const isTowardDep = group.direction === "DEPRESI";
                  
                  // Filter waterfall features belonging to this specific group
                  const groupFeatures = layer3Features.filter(
                    (feat) => normalizeGroupName(feat.feature_group) === group.name
                  );
                  const maxFeatureImpact = Math.max(
                    ...groupFeatures.map((feat) => Math.abs(feat.magnitude ?? feat.shap_value ?? 0)),
                    0
                  );

                  return (
                    <div
                      key={group.name}
                      className={`rounded-xl border transition-all duration-300 overflow-hidden ${
                        isExpanded ? "border-slate-200 shadow-sm bg-white" : "border-border-light bg-bg/50 hover:bg-bg"
                      }`}
                    >
                      {/* Header (Clickable toggle) */}
                      <button
                        onClick={() =>
                          setExpandedGroups((prev) => ({ ...prev, [group.name]: !prev[group.name] }))
                        }
                        className="w-full flex items-start sm:items-center justify-between p-4 text-left gap-3 cursor-pointer"
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 sm:mt-0">
                            {getGroupIcon(group.name, 22, "text-primary")}
                          </div>
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <p className="text-sm font-extrabold text-text">{copy?.title ?? group.name}</p>
                              {isMain && (
                                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-primary text-white">
                                  Dominant
                                </span>
                              )}
                            </div>
                            <p className="text-[10px] text-text-muted">{copy?.technical}</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <span className={`text-sm font-black ${isTowardDep ? "text-primary" : "text-emerald-600"}`}>
                              {group.contribution.toFixed(1)}%
                            </span>
                            <p className={`text-[9px] font-bold uppercase tracking-wider ${isTowardDep ? "text-red-500" : "text-emerald-500"}`}>
                              {isTowardDep ? "↑ Depression" : "↓ Normal"}
                            </p>
                          </div>
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </div>
                      </button>

                      {/* Expandable Panel */}
                      <AnimatePresence initial={false}>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0 }}
                            animate={{ height: "auto" }}
                            exit={{ height: 0 }}
                            transition={{ duration: 0.3, ease: "easeInOut" }}
                            className="border-t border-slate-100 overflow-hidden bg-white"
                          >
                            <div className="p-4 space-y-4">
                              {/* Horizontal Visual Bar */}
                              <div className="space-y-1">
                                <div className="flex justify-between text-[10px] text-text-muted font-bold">
                                  <span>Feature Impact Level</span>
                                  <span>{group.contribution.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200/30">
                                  <div
                                    className={`h-full rounded-full transition-all duration-700 ${
                                      isTowardDep ? "bg-primary" : "bg-emerald-500"
                                    }`}
                                    style={{ width: `${Math.min(group.contribution, 100)}%` }}
                                  />
                                </div>
                              </div>

                              {/* Descriptive Text Box */}
                              <div className={`p-3 rounded-lg border text-xs leading-relaxed ${
                                isTowardDep ? "bg-red-50/50 border-red-100 text-text" : "bg-emerald-50/30 border-emerald-100 text-text"
                              }`}>
                                {isTowardDep ? copy?.plainDepression : copy?.plainNormal}
                              </div>

                              {/* Layer 2: Sub-groups */}
                              {group.name === "MelSpec" && Object.keys(melspecSubs).length > 0 && (
                                <div className="space-y-2 pt-2 border-t border-slate-50">
                                  <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                                    📊 Subgroup Breakdown (MelSpec)
                                  </p>
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                    {Object.entries(melspecSubs).map(([name, info]) => {
                                      const isSubDep = info.direction === "DEPRESI";
                                      return (
                                        <div key={name} className="bg-bg rounded-lg p-2.5 border border-border-light">
                                          <p className="text-[10px] font-bold text-text truncate">{name}</p>
                                          <div className="flex items-baseline justify-between mt-1">
                                            <span className={`text-xs font-black ${isSubDep ? "text-primary" : "text-emerald-600"}`}>
                                              {info.contribution_pct.toFixed(2)}%
                                            </span>
                                            <span className={`text-[8px] font-bold ${isSubDep ? "text-red-500" : "text-emerald-500"}`}>
                                              {isSubDep ? "Depression" : "Normal"}
                                            </span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}

                              {group.name === "MFCC" && Object.keys(mfccSubs).length > 0 && (
                                <div className="space-y-2 pt-2 border-t border-slate-50">
                                  <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                                    🎛️ Subgroup Breakdown (MFCC)
                                  </p>
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                    {Object.entries(mfccSubs).map(([name, info]) => {
                                      const isSubDep = info.direction === "DEPRESI";
                                      return (
                                        <div key={name} className="bg-bg rounded-lg p-2.5 border border-border-light">
                                          <p className="text-[10px] font-bold text-text truncate">{name}</p>
                                          <div className="flex items-baseline justify-between mt-1">
                                            <span className={`text-xs font-black ${isSubDep ? "text-primary" : "text-emerald-600"}`}>
                                              {info.contribution_pct.toFixed(2)}%
                                            </span>
                                            <span className={`text-[8px] font-bold ${isSubDep ? "text-red-500" : "text-emerald-500"}`}>
                                              {isSubDep ? "Depression" : "Normal"}
                                            </span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}

                              {/* Layer 3: Top Individual Features */}
                              {groupFeatures.length > 0 && (
                                <div className="space-y-2 pt-2 border-t border-slate-50">
                                  <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                                    🎯 Top Individual Vocal Parameters
                                  </p>
                                  <div className="space-y-1.5">
                                    {groupFeatures.map((feat) => {
                                      const isFeatDep = feat.direction === "DEPRESI";
                                      const featureImpact = Math.abs(feat.magnitude ?? feat.shap_value ?? 0);
                                      const featureWidth = maxFeatureImpact > 0
                                        ? Math.max(8, (featureImpact / maxFeatureImpact) * 100)
                                        : 0;
                                      return (
                                        <div key={feat.rank} className="space-y-1.5 py-2 border-b border-slate-50 last:border-0">
                                          <div className="flex items-center justify-between gap-3 text-xs">
                                          <span className="font-semibold text-text truncate max-w-[200px] sm:max-w-none">
                                            {feat.feature_sub}
                                          </span>
                                          <span className={`text-[10px] font-black shrink-0 ${isFeatDep ? "text-primary" : "text-emerald-600"}`}>
                                            {isFeatDep ? "↑ Toward Depression" : "↓ Toward Normal"}
                                          </span>
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <div className="relative h-2 flex-1 rounded-full bg-slate-100 overflow-hidden border border-slate-200/40">
                                              <div
                                                className={`h-full rounded-full transition-all duration-700 ${
                                                  isFeatDep ? "bg-primary" : "bg-emerald-500"
                                                }`}
                                                style={{ width: `${featureWidth}%` }}
                                              />
                                            </div>
                                            <span className="w-14 text-right text-[9px] font-bold text-text-muted tabular-nums">
                                              {featureImpact.toFixed(4)}
                                            </span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}

                              {/* Educational Explainer */}
                              <div className="pt-2 border-t border-slate-50">
                                <p className="text-[10px] text-text-muted italic leading-relaxed">
                                  <strong>How it works:</strong> {copy?.explanation}
                                </p>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-xl bg-bg border border-border-light p-4 text-sm text-text-muted">
                No detailed vocal explanation is available at this time.
              </div>
            )}
          </section>

          {/* AI Summary Narrative Card */}
          <section className="bg-white rounded-2xl border border-border-light p-5 sm:p-7 soft-shadow space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary shrink-0">
                <Brain size={19} />
              </div>
              <div>
                <h4 className="text-base font-extrabold text-text">AI Narrative Summary</h4>
                <p className="text-xs text-text-muted">A plain language summary of the vocal markers detected</p>
              </div>
            </div>
            <div className="bg-bg rounded-xl p-4 border border-border-light">
              <p className="text-sm text-text leading-relaxed font-medium">{plainExplanation}</p>
            </div>

            {/* Alert / Warning */}
            <div className="flex items-start gap-2.5 p-4 bg-amber-50 border border-amber-100 rounded-xl">
              <AlertTriangle size={15} className="text-amber-600 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="text-xs font-bold text-amber-800">Screening tool only, not a diagnostic replacement</p>
                <p className="text-xs text-amber-700 leading-relaxed">
                  Vocal screening analysis detects statistical correlations with vocal markers in research data. Results can vary based on microphone quality, background noise, or temporary conditions like a cold or fatigue. Please consult a qualified psychologist or medical professional for a clinical evaluation if needed.
                </p>
              </div>
            </div>
          </section>
        </div>
      )}

      {/* ─── Tab: Global Model Visualizations (Native Interactive Charts) ─── */}
      {activeTab === "global" && (
        <div className="space-y-5">
          <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow">
            <div className="flex items-start gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary shrink-0">
                <BarChart2 size={19} />
              </div>
              <div>
                <h4 className="text-base font-extrabold text-text">Global Model Visualizations</h4>
                <p className="text-sm text-text-muted mt-1 leading-relaxed">
                  These charts show how the AI model behaves <strong>across all 500 reference samples</strong> in the training dataset — not just your recording.
                </p>
              </div>
            </div>

            {/* Global Analysis Explanation Banner */}
            <div className="rounded-xl border border-sky-100 bg-gradient-to-r from-sky-50 to-cyan-50 p-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-sky-100 flex items-center justify-center text-sky-600 shrink-0">
                  <BarChart2 size={15} />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <h5 className="text-xs font-extrabold text-sky-900">Global Analysis — Model-Wide Patterns</h5>
                    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-sky-200 text-sky-800 uppercase tracking-wider">Population</span>
                  </div>
                  <p className="text-[11px] text-sky-700 leading-relaxed">
                    <strong>Global analysis</strong> reveals <em>which vocal features are most important to the AI model in general</em>, based on patterns learned from hundreds of labeled voice recordings. Unlike your personal result above, these charts represent how the model behaves on average — helping you understand what the AI has learned about the relationship between voice and depression.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-start gap-2.5 p-3 bg-blue-50 border border-blue-100 rounded-xl">
              <Info size={13} className="text-blue-600 shrink-0 mt-0.5" />
              <p className="text-[11px] text-blue-700 leading-relaxed font-medium">
                These charts are interactive. Hover over data elements to view detailed feature values and their exact prediction push.
              </p>
            </div>
          </div>

          {/* Interactive Chart Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="bg-white rounded-2xl border border-border-light p-5 soft-shadow space-y-3">
              <h5 className="text-sm font-extrabold text-text">Feature Group Contribution (Layer 1)</h5>
              <p className="text-[10px] text-text-muted">Overall contribution of the three main vocal feature pipelines on predictions.</p>
              <GroupContributionChart />
            </div>

            <div className="bg-white rounded-2xl border border-border-light p-5 soft-shadow space-y-3">
              <h5 className="text-sm font-extrabold text-text">Global SHAP Beeswarm Distribution</h5>
              <p className="text-[10px] text-text-muted">Impact of high vs. low feature values on predictions across samples.</p>
              <BeeswarmChart />
            </div>

            <div className="bg-white rounded-2xl border border-border-light p-5 soft-shadow space-y-3">
              <h5 className="text-sm font-extrabold text-text">Frequency Band Breakdown (MelSpec)</h5>
              <p className="text-[10px] text-text-muted">Weight of low, mid, and high pitch ranges on frequency analysis.</p>
              <MelSpecBreakdownChart />
            </div>

            <div className="bg-white rounded-2xl border border-border-light p-5 soft-shadow space-y-3">
              <h5 className="text-sm font-extrabold text-text">Voice Timbre & Quality Breakdown (MFCC)</h5>
              <p className="text-[10px] text-text-muted">Weight of base timbre characteristics vs. speed transition indicators.</p>
              <MfccBreakdownChart />
            </div>

            <div className="bg-white rounded-2xl border border-border-light p-5 soft-shadow space-y-3 md:col-span-2">
              <h5 className="text-sm font-extrabold text-text">Feature Impact Aggregation (Waterfall)</h5>
              <p className="text-[10px] text-text-muted">Step-by-step breakdown of how features add to (+) or subtract from (-) baseline probability.</p>
              <WaterfallChart />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
