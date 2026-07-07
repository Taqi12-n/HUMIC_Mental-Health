"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Activity, Play, Pause, Brain, Clock, AlertCircle,
  RefreshCw, Info, BarChart2, CheckCircle2, Cpu, Target,
  Layers, TrendingUp, Volume2, VolumeX, ExternalLink, Zap,
  Mic, Sparkles
} from "lucide-react";
import Link from "next/link";
import { getApiUrl } from "@/utils/api";

// ─── Types ─────────────────────────────────────────────────────────────────────

type ModelInfo = {
  name: string;
  source?: string;
  xaiSource?: string;
  scenario: string;
  depressionProbability: number;
  threshold: number;
  testF1: string;
};

type SegmentInfo = {
  totalSegments: number;
  threshold: number;
  votes?: {
    NORMAL?: number;
    DEPRESI?: number;
  };
  segmentDetail?: Array<{
    segment_index: number;
    duration_sec: number;
    pred_label: string;
    prob_depresi: number;
  }>;
};

type AudioResultData = {
  id: string;
  audioInfo?: {
    audioUrl?: string;
  };
};

type AnalysisResultData = AudioResultData & {
  filename?: string;
  date?: string;
  primaryDetection: string;
  confidence: number;
  metrics: {
    depression: number;
    normal: number;
  };
  audioInfo?: {
    duration?: string;
    avgPitch?: string;
    energyLevel?: string;
    signalQuality?: string;
    audioUrl?: string;
  };
  recommendation?: {
    title?: string;
    text?: string;
  };
  modelInfo?: ModelInfo;
  segmentInfo?: SegmentInfo;
  performance?: {
    testF1?: string;
    scenario?: string;
    threshold?: number;
  };
};

// ─── Helpers ───────────────────────────────────────────────────────────────────

const resolveAudioUrl = (data: AudioResultData) => {
  const rawAudioUrl = data.audioInfo?.audioUrl || `/api/audio/${data.id}`;
  return rawAudioUrl.startsWith("http") ? rawAudioUrl : getApiUrl(rawAudioUrl);
};

// ─── Model Metric Card ─────────────────────────────────────────────────────────

function ModelMetricsCard({ modelInfo, segmentInfo, performance }: {
  modelInfo?: any;
  segmentInfo?: any;
  performance?: any;
} = {}) {
  return (
    <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
            <Cpu size={20} />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-extrabold text-text">
              How Analysis Works
            </h4>
            <p className="text-xs text-text-muted">
              How your voice recording is evaluated from audio input to the final analysis report
            </p>
          </div>
        </div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-100 text-[10px] font-bold text-emerald-700">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          Active
        </div>
      </div>

      {/* Feature pipeline explainer */}
      <div className="bg-bg rounded-xl p-4 border border-border-light space-y-3">
        <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
          Your Voice's Journey into AI Analysis
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mt-2">
          {[
            {
              step: "1",
              icon: <Mic size={18} className="text-primary" />,
              title: "Voice Recording",
              desc: "AI captures speech frequencies, rhythm, pauses, and pitch variations from your recording."
            },
            {
              step: "2",
              icon: <BarChart2 size={18} className="text-primary" />,
              title: "Feature Extraction",
              desc: "Vocal patterns, tone quality, and speech properties are extracted to detect subtle emotional markers.",
              showInfo: true
            },
            {
              step: "3",
              icon: <Brain size={18} className="text-primary" />,
              title: "Model Evaluation",
              desc: "Machine learning algorithms compare your voice patterns with thousands of clinical records."
            },
            {
              step: "4",
              icon: <Sparkles size={18} className="text-primary" />,
              title: "Analysis Report",
              desc: "An analysis showing depression indicators and vocal markers is visualised transparently."
            }
          ].map((item) => (
            <div key={item.step} className="bg-white border border-slate-100/50 rounded-xl p-4 transition-all duration-300 hover:shadow-md hover:border-primary/20 space-y-1.5 relative group">
              <div className="flex items-center gap-2">
                {item.icon}
                <span className="text-[10px] font-bold uppercase tracking-wider text-primary">Step {item.step}</span>
              </div>
              <div className="flex items-center gap-1">
                <h5 className="text-xs font-bold text-text">{item.title}</h5>
                {item.showInfo && (
                  <div className="cursor-pointer text-text-light hover:text-primary transition-colors">
                    <Info size={12} />
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 bg-slate-900 text-white text-[10px] p-2.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 leading-relaxed shadow-lg">
                      <strong>Extracted features:</strong>
                      <ul className="list-disc list-inside mt-1 space-y-0.5">
                        <li>Mel Spectrogram (Pitch)</li>
                        <li>MFCC (Vocal timbre)</li>
                        <li>Wav2Vec (Speech flow)</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
              <p className="text-[10px] text-text-muted leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2.5 p-3 bg-amber-50 border border-amber-100 rounded-xl">
        <Info size={14} className="text-amber-600 shrink-0 mt-0.5" />
        <p className="text-[10px] text-amber-700 leading-relaxed">
          <strong>Research Screening Tool.</strong> This AI tool screens for vocal indicators associated with depression markers. It is not a clinical diagnosis. Please consult a mental health professional for further medical advice.
        </p>
      </div>
    </div>
  );
}

// ─── Results Content ───────────────────────────────────────────────────────────

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const resultId = searchParams?.get("id");

  const [data, setData] = useState<AnalysisResultData | null>(null);
  const [resultError, setResultError] = useState("");
  const [loading, setLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0.0);
  const [totalDuration, setTotalDuration] = useState(45.0);
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);
  const [audioError, setAudioError] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Fetch results from backend
  useEffect(() => {
    const activeId = resultId || localStorage.getItem("mindvoice_active_result_id");

    if (!activeId) {
      router.push("/#upload");
      return;
    }

    if (activeId === "fallback-mock-id") {
      localStorage.removeItem("mindvoice_active_result_id");
      router.push("/#upload");
      return;
    }

    const fetchResult = async () => {
      try {
        const response = await fetch(getApiUrl(`/api/results/${activeId}`));
        if (!response.ok) throw new Error("Result not found");
        const json = (await response.json()) as AnalysisResultData;
        setData(json);
        if (json.audioInfo?.duration) {
          const parsed = parseFloat(json.audioInfo.duration);
          if (!isNaN(parsed)) {
            setTotalDuration(parsed);
          }
        }
      } catch (err) {
        console.warn("Backend fetch failed.", err);
        setResultError(
          "Analysis result could not be loaded. Please upload and analyze the audio again."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [resultId, router]);

  // Decode audio to extract waveform peaks
  useEffect(() => {
    if (loading || !data) return;

    const audioUrl = resolveAudioUrl(data);

    const loadWaveform = async () => {
      try {
        const response = await fetch(audioUrl, { credentials: "omit" });
        if (!response.ok) throw new Error("Failed to fetch audio file");
        const arrayBuffer = await response.arrayBuffer();
        const AudioContextClass =
          window.AudioContext ||
          (window as typeof window & { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext;
        if (!AudioContextClass) throw new Error("AudioContext is not supported");
        const audioCtx = new AudioContextClass();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        const rawData = audioBuffer.getChannelData(0);
        const samples = 64;
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];
        for (let i = 0; i < samples; i++) {
          const blockStart = blockSize * i;
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[blockStart + j]);
          }
          filteredData.push(sum / blockSize);
        }

        const max = Math.max(...filteredData) || 1;
        const normalized = filteredData.map((val) => (val / max) * 0.8 + 0.15);
        setWaveformPeaks(normalized);
      } catch (err) {
        console.warn("Could not decode audio data for waveform", err);
        const peaks = Array.from({ length: 64 }, (_, idx) => {
          const seed = data.id
            .split("")
            .reduce((acc: number, char: string) => acc + char.charCodeAt(0), 0);
          return Math.abs(Math.sin(idx * 0.1 + seed)) * 0.75 + 0.15;
        });
        setWaveformPeaks(peaks);
      }
    };

    loadWaveform();
  }, [data, loading]);

  useEffect(() => {
    if (audioRef.current && data) {
      audioRef.current.load();
    }
  }, [data]);

  const handlePlayToggle = () => {
    if (!audioRef.current) return;
    setAudioError(false);
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => {
          console.error("Audio play failed", err);
          setAudioError(true);
        });
    }
  };

  const handleWaveformSeek = (index: number) => {
    if (!audioRef.current || totalDuration <= 0 || waveformPeaks.length === 0) return;
    const time = (index / Math.max(waveformPeaks.length - 1, 1)) * totalDuration;
    const seekTime = Math.min(time, totalDuration);
    audioRef.current.currentTime = seekTime;
    setCurrentTime(seekTime);
    audioRef.current
      .play()
      .then(() => setIsPlaying(true))
      .catch((err) => {
        console.error("Seek play failed", err);
        setAudioError(true);
      });
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextVal = parseFloat(e.target.value);
    setCurrentTime(nextVal);
    if (audioRef.current) {
      audioRef.current.currentTime = nextVal;
    }
  };

  const handleAnalyzeAnother = () => {
    localStorage.removeItem("mindvoice_active_result_id");
    window.dispatchEvent(new Event("storage"));
    router.push("/");
  };

  const confidenceLabel = (value: number) =>
    value >= 75 ? "High Confidence" : value >= 50 ? "Medium Confidence" : "Low Confidence";

  const confidenceSummary = (value: number) =>
    value >= 75
      ? "The AI is quite confident in this result."
      : value >= 50
        ? "The AI has moderate confidence in this result."
        : "The AI's confidence is lower, so treat this as an early signal.";

  const formatTime = (secs: number) => {
    const roundedSecs = Math.floor(secs);
    const minutes = Math.floor(roundedSecs / 60);
    const seconds = roundedSecs % 60;
    return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
  };

  const playProgress = totalDuration > 0 ? currentTime / totalDuration : 0;

  if (loading) {
    return (
      <div className="min-h-screen bg-bg flex flex-col items-center justify-center p-4">
        <RefreshCw className="animate-spin text-primary mb-4" size={32} />
        <p className="text-text-muted font-medium">Fetching analysis report...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-bg flex flex-col items-center justify-center p-4 text-center">
        <div className="w-14 h-14 rounded-2xl bg-red-50 text-primary flex items-center justify-center mb-4">
          <AlertCircle size={28} />
        </div>
        <h2 className="text-xl font-extrabold text-text mb-2">
          Analysis Result Unavailable
        </h2>
        <p className="text-sm text-text-muted max-w-md leading-relaxed mb-6">
          {resultError ||
            "The analysis result could not be loaded. Please upload the audio again."}
        </p>
        <button
          onClick={() => {
            localStorage.removeItem("mindvoice_active_result_id");
            router.push("/#upload");
          }}
          className="px-6 py-3 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
        >
          Upload Again
        </button>
      </div>
    );
  }

  // Circular gauge
  const radius = 58;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset =
    circumference - (data.confidence / 100) * circumference;

  const isDepression =
    data.primaryDetection === "Depression" ||
    data.primaryDetection === "DEPRESI";
  const activeColor = isDepression
    ? { stroke: "#E91E63", text: "text-primary", bg: "bg-red-500" }
    : { stroke: "#10B981", text: "text-green-500", bg: "bg-green-500" };

  const predictionText = isDepression
    ? "Your recording shows more features similar to the depression pattern."
    : "Your recording shows more features similar to a typical speech pattern.";

  const predictionTone = isDepression
    ? "The AI detected voice clues that more often appear in depression-related patterns."
    : "The AI did not find strong voice clues typically linked to depression.";

  const depressionPercent = data.metrics?.depression ?? data.confidence;

  const probabilityText = isDepression
    ? "This result leans toward the depression category."
    : "This result leans toward the typical speech category.";

  const depressionSegment = Math.min(7, Math.floor(depressionPercent / 12.5));
  const depressionSegmentLabel = [
    "0 - 12.5%",
    "12.5 - 25%",
    "25 - 37.5%",
    "37.5 - 50%",
    "50 - 62.5%",
    "62.5 - 75%",
    "75 - 87.5%",
    "87.5 - 100%",
  ][depressionSegment];

  const segmentRecommendation = [
    {
      title: "Very low depression signal",
      summary:
        "Your score is very low. Continue keeping a healthy routine and stay mindful of your mood.",
      do: [
        "Keep a balanced sleep schedule.",
        "Stay active with gentle movement.",
        "Keep a simple mood check every few days.",
      ],
      dont: [
        "Don't ignore any sudden changes in your mood.",
        "Don't avoid talking with someone if you feel uneasy.",
      ],
    },
    {
      title: "Low depression signal",
      summary:
        "Your score is low, but staying aware and practicing self-care helps keep it there.",
      do: [
        "Continue regular social contact.",
        "Practice relaxation techniques like breathing or stretching.",
        "Write down small positive moments each day.",
      ],
      dont: [
        "Don't let stress build up without taking a short break.",
        "Don't withdraw from activities that usually feel good.",
      ],
    },
    {
      title: "Mild depression signal",
      summary:
        "Your score is mildly elevated. Focus on self-care, rest, and sharing how you feel with someone trusted.",
      do: [
        "Talk with a friend, family member, or support person.",
        "Keep a simple mood journal.",
        "Make time for healthy sleep and movement.",
      ],
      dont: [
        "Don't push yourself too hard if you feel tired.",
        "Don't ignore repeated low moods.",
      ],
    },
    {
      title: "Moderate depression signal",
      summary:
        "Your score is moderate. It is helpful to pay closer attention to your feelings and seek support if they continue.",
      do: [
        "Keep a daily check-in on mood and energy.",
        "Share what you feel with a trusted person.",
        "Use small routines to stay grounded.",
      ],
      dont: [
        "Don't isolate yourself when you are feeling down.",
        "Don't let sleep or eating patterns slip too far.",
      ],
    },
    {
      title: "Elevated depression signal",
      summary:
        "Your score is elevated. Try to connect with support and consider whether a professional check-in is needed.",
      do: [
        "Talk with someone you trust about how you feel.",
        "Keep up small self-care habits like hydration and rest.",
        "Notice if low mood lasts more than a few days.",
      ],
      dont: [
        "Don't dismiss continued sadness as just a bad day.",
        "Don't use alcohol or other quick fixes to cope.",
      ],
    },
    {
      title: "High depression signal",
      summary:
        "Your score is high. It is a good idea to get help from a professional or counselor soon.",
      do: [
        "Reach out to a mental health professional or counselor.",
        "Share this result with a trusted person.",
        "Keep a short record of your mood and sleep.",
      ],
      dont: [
        "Don't wait until you feel much worse.",
        "Don't cope alone if these feelings are persistent.",
      ],
    },
    {
      title: "Very high depression signal",
      summary:
        "Your score is very high. Please consider professional support and ask for help from someone close.",
      do: [
        "Contact a clinician, counselor, or trusted support person.",
        "Use emergency or crisis resources if needed.",
        "Keep up basic self-care like eating and resting.",
      ],
      dont: [
        "Don't ignore strong feelings of sadness or hopelessness.",
        "Don't put off seeking help because you feel unsure.",
      ],
    },
    {
      title: "Extremely high depression signal",
      summary:
        "Your score is extremely high. Immediate help and professional support are strongly advised.",
      do: [
        "Reach out to medical or mental health services right away.",
        "Tell a trusted person how you are feeling.",
        "Use crisis services if you feel unsafe.",
      ],
      dont: [
        "Don't delay seeking professional help.",
        "Don't cope alone if your feelings are overwhelming.",
      ],
    },
  ][depressionSegment];

  const recommendationItems = [
    {
      text: `Depression score: ${depressionPercent.toFixed(1)}%`,
      type: "score",
      label: "",
    },
    {
      label: segmentRecommendation.title,
      text: segmentRecommendation.summary,
      type: "summary",
    },
  ];
  const recommendationDo = segmentRecommendation.do;
  const recommendationDont = segmentRecommendation.dont;

  const audioSrc = resolveAudioUrl(data);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10 space-y-6 sm:space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-text-muted mb-2">
          <span>Analysis summary</span>
          <span>•</span>
          <span>{data.date}</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-text mb-1">
          Result overview
        </h2>
        <p className="text-text-muted text-sm sm:text-base">
          A clear summary of what the AI found in your voice recording.
        </p>
      </div>

      {/* Primary Detection Card */}
      <div className="bg-white rounded-2xl soft-shadow border border-border-light p-5 sm:p-7 grid md:grid-cols-3 gap-6 sm:gap-8 items-center">
        <div className="md:col-span-2 space-y-4">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-50 text-xs font-semibold text-primary border border-red-100/50">
            <Activity size={12} />
            Result summary
          </div>
          <div>
            <h3 className="text-2xl sm:text-3xl font-black text-text tracking-tight mb-2">
              {data.primaryDetection}
            </h3>
            <p className="text-sm text-text-muted leading-relaxed">
              {predictionText}
            </p>
            <p className="text-sm text-text-muted leading-relaxed mt-2">
              {predictionTone}
            </p>
          </div>

          <div className="space-y-1.5 pt-2">
            <div className="flex justify-between text-xs font-bold text-text-muted">
              <span>Confidence Score</span>
              <span className={activeColor.text}>{data.confidence}%</span>
            </div>
            <div className="w-full bg-bg rounded-full h-2.5 overflow-hidden">
              <div
                className="gradient-bg h-full rounded-full transition-all duration-1000"
                style={{ width: `${data.confidence}%` }}
              />
            </div>
            <p className="text-[11px] text-text-muted">{confidenceSummary(data.confidence)}</p>
          </div>

          <div className="flex items-start gap-2.5 p-3.5 bg-red-50/50 border border-red-100/50 rounded-xl">
            <Info size={16} className="text-primary shrink-0 mt-0.5" />
            <p className="text-[11px] sm:text-xs text-text-muted leading-relaxed">
              <strong className="text-text">Attention: </strong> This
              summary is meant to help you understand your recording and is not
              a substitute for medical advice.
            </p>
          </div>
        </div>

        {/* Circular Gauge */}
        <div className="flex flex-col items-center justify-center p-4">
          <div className="relative flex items-center justify-center w-[140px] h-[140px]">
            <svg
              viewBox="0 0 140 140"
              className="w-full h-full transform -rotate-90 select-none"
            >
              <circle
                stroke="#F1F5F9"
                fill="transparent"
                strokeWidth="12"
                r={radius}
                cx="70"
                cy="70"
              />
              <motion.circle
                stroke={activeColor.stroke}
                fill="transparent"
                strokeWidth="12"
                strokeDasharray={`${circumference}`}
                style={{ strokeDashoffset }}
                r={radius}
                cx="70"
                cy="70"
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset }}
                transition={{ duration: 1, ease: "easeOut" }}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute text-center">
              <p className="text-3xl font-black text-text leading-none">
                {data.confidence}%
              </p>
              <p className="text-[10px] text-text-light font-bold uppercase tracking-wider mt-1">
                Confidence
              </p>
            </div>
          </div>
          <p className="text-xs text-text-muted text-center mt-3 font-medium">
            {confidenceLabel(data.confidence)}
          </p>
        </div>
      </div>

      {/* Depression / Normal metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-border-light p-4 soft-shadow">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-bold text-text-muted">Depression</span>
            <Activity size={14} className="text-primary" />
          </div>
          <h4 className="text-xl sm:text-2xl font-black text-text mb-2">
            {data.metrics.depression}%
          </h4>
          <div className="w-full bg-bg h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-primary h-full rounded-full"
              style={{ width: `${data.metrics.depression}%` }}
            />
          </div>
        </div>

        <div className="bg-white rounded-xl border border-border-light p-4 soft-shadow">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-bold text-text-muted">Normal</span>
            <Activity size={14} className="text-green-500" />
          </div>
          <h4 className="text-xl sm:text-2xl font-black text-text mb-2">
            {data.metrics.normal}%
          </h4>
          <div className="w-full bg-bg h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-green-500 h-full rounded-full"
              style={{ width: `${data.metrics.normal}%` }}
            />
          </div>
        </div>
      </div>
      <p className="text-sm text-text-muted mt-3">
        {probabilityText}
      </p>

      {/* ── Model Performance Metrics Card ── */}
      <ModelMetricsCard
        modelInfo={data.modelInfo}
        segmentInfo={data.segmentInfo}
        performance={data.performance}
      />

      {/* Audio Information */}
      <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-4">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
            <Clock size={20} />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-extrabold text-text">
              Audio details
            </h4>
            <p className="text-xs text-text-muted">
              Information about the uploaded recording.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <div className="bg-bg rounded-2xl p-4 border border-border-light">
            <p className="text-[10px] uppercase tracking-wider text-text-light font-bold mb-1">
              Duration
            </p>
            <p className="text-sm font-semibold text-text">
              {formatTime(totalDuration)}
            </p>
          </div>
          {data.audioInfo?.avgPitch && (
            <div className="bg-bg rounded-2xl p-4 border border-border-light">
              <p className="text-[10px] uppercase tracking-wider text-text-light font-bold mb-1">
                Average pitch
              </p>
              <p className="text-sm font-semibold text-text">
                {data.audioInfo.avgPitch}
              </p>
            </div>
          )}
          {data.audioInfo?.energyLevel && (
            <div className="bg-bg rounded-2xl p-4 border border-border-light">
              <p className="text-[10px] uppercase tracking-wider text-text-light font-bold mb-1">
                Energy
              </p>
              <p className="text-sm font-semibold text-text">
                {data.audioInfo.energyLevel}
              </p>
            </div>
          )}
          {data.audioInfo?.signalQuality && (
            <div className="bg-bg rounded-2xl p-4 border border-border-light">
              <p className="text-[10px] uppercase tracking-wider text-text-light font-bold mb-1">
                Signal quality
              </p>
              <p className="text-sm font-semibold text-text">
                {data.audioInfo.signalQuality}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Recommendation */}
      <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-6">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
            <Brain size={20} />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-extrabold text-text">
              What should you do next?
            </h4>
            <p className="text-xs text-text-muted">Recommendations based on your depression score.</p>
          </div>
        </div>

        <div className="space-y-3">
          {recommendationItems.map((item) => (
            <div key={item.label} className="rounded-2xl bg-bg border border-border-light p-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <p className="text-xs uppercase tracking-wide text-text-muted font-semibold">
                  {item.type === "score" ? "Score" : "Summary"}
                </p>
                <span className="text-xs text-text-muted">{item.label}</span>
              </div>
              <p className="text-sm text-text leading-relaxed">{item.text}</p>
            </div>
          ))}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-2xl bg-green-50 border border-green-200 p-4">
              <h5 className="text-sm font-bold text-text mb-2">What to do</h5>
              <ul className="space-y-2 list-disc list-inside text-sm text-text-muted">
                {recommendationDo.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="rounded-2xl bg-red-50 border border-red-200 p-4">
              <h5 className="text-sm font-bold text-text mb-2">What not to do</h5>
              <ul className="space-y-2 list-disc list-inside text-sm text-text-muted">
                {recommendationDont.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className="flex items-start gap-2.5 p-3.5 bg-bg border border-border-light rounded-xl">
          <Info size={16} className="text-primary shrink-0 mt-0.5" />
          <p className="text-xs text-text-muted leading-relaxed">
            This is a preliminary screen only and does not replace professional help.
          </p>
        </div>
      </div>

      {/* Audio Player */}
      <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border-light">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
              <Volume2 size={20} />
            </div>
            <div>
              <h4 className="text-sm sm:text-base font-extrabold text-text">
                Play your recording
              </h4>
              <p className="text-xs text-text-muted mt-0.5">
                Listen back to the audio that was analyzed.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handlePlayToggle}
              className="inline-flex items-center justify-center gap-2 rounded-full bg-primary px-5 py-2.5 text-white text-sm font-semibold shadow hover:opacity-90 transition-opacity"
            >
              {isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" />}
              {isPlaying ? "Pause" : "Play"}
            </button>
            <span className="text-xs text-text-muted font-mono">
              {formatTime(currentTime)} / {formatTime(totalDuration)}
            </span>
          </div>
        </div>

        {/* Hidden audio element */}
        <audio
          ref={audioRef}
          src={audioSrc}
          preload="auto"
          crossOrigin="anonymous"
          onTimeUpdate={() => {
            if (audioRef.current) {
              setCurrentTime(audioRef.current.currentTime);
            }
          }}
          onLoadedMetadata={() => {
            if (audioRef.current) {
              setTotalDuration(audioRef.current.duration || 45.0);
            }
          }}
          onCanPlay={() => {
            if (audioRef.current) {
              audioRef.current.volume = 1;
              audioRef.current.muted = false;
            }
          }}
          onEnded={() => setIsPlaying(false)}
          onError={(event) => {
            console.warn("Audio element error:", event);
            setAudioError(true);
            setIsPlaying(false);
          }}
          className="sr-only"
        />

        {/* Audio error state */}
        {audioError && (
          <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-100 rounded-xl">
            <VolumeX size={16} className="text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-2 flex-1">
              <p className="text-xs font-semibold text-amber-800">
                Browser playback unavailable. Use the native player below:
              </p>
              <audio
                src={audioSrc}
                controls
                className="w-full h-9"
                style={{ borderRadius: 8 }}
              />
            </div>
          </div>
        )}

        {/* Seek slider */}
        {!audioError && (
          <div className="space-y-2">
            <input
              type="range"
              min={0}
              max={totalDuration}
              step={0.1}
              value={currentTime}
              onChange={handleSliderChange}
              className="w-full accent-primary cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-text-muted font-mono">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(totalDuration)}</span>
            </div>
          </div>
        )}

        {/* Waveform Visualization */}
        <div className="bg-bg rounded-xl p-4 flex items-end justify-between h-20 gap-[2px] select-none">
          {(waveformPeaks.length > 0
            ? waveformPeaks
            : Array.from({ length: 64 }).map(() => 0.15)
          ).map((peak, idx) => {
            const h = peak * 90;
            const isPlayed = idx / Math.max(waveformPeaks.length - 1, 1) < playProgress;
            return (
              <div
                key={idx}
                onClick={() => handleWaveformSeek(idx)}
                className={`w-[3px] cursor-pointer rounded-full transition-all duration-100 ${isPlayed ? "bg-primary opacity-90" : "bg-slate-200 hover:bg-slate-300"}`}
                style={{ height: `${h}%` }}
              />
            );
          })}
        </div>

        <p className="text-[11px] text-text-muted text-center">
          Click any point on the waveform to jump to that moment
        </p>
      </div>

      {/* Navigation Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 pt-4">
        <Link
          href={
            (resultId || data?.id)
              ? `/ai-insight?id=${resultId || data?.id}`
              : "/ai-insight"
          }
          className="flex-1 text-center py-3.5 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
        >
          View AI Insights (XAI)
        </Link>
        <button
          onClick={handleAnalyzeAnother}
          className="flex-1 text-center py-3.5 rounded-full border border-border bg-white text-text font-bold text-sm hover:bg-bg transition-colors"
        >
          Analyze Another Audio
        </button>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  const [activeId] = useState<string | null>(() =>
    typeof window !== "undefined"
      ? localStorage.getItem("mindvoice_active_result_id")
      : null
  );

  return (
    <>
      {/* Sticky header */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-navbar shadow-sm border-b border-white/50 h-16 sm:h-20 flex items-center">
        <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full gradient-bg flex items-center justify-center shadow-md">
              <div className="w-4 h-4 bg-white/30 rounded-full" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold text-text leading-tight">
                MindVoice AI
              </h1>
              <p className="text-[10px] sm:text-xs text-text-muted leading-tight">
                Telkom University
              </p>
            </div>
          </Link>

          <div className="flex items-center gap-1">
            <Link
              href="/"
              className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-text-muted hover:text-text"
            >
              Home
            </Link>
            <span className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-white gradient-bg shadow-sm">
              Results
            </span>
            <Link
              href={activeId ? `/ai-insight?id=${activeId}` : "/ai-insight"}
              className="px-4 py-2 rounded-full text-xs sm:text-sm font-medium text-text-muted hover:text-text"
            >
              AI Insights
            </Link>
          </div>
        </div>
      </nav>

      {/* Main Container */}
      <main className="min-h-screen pt-20 sm:pt-24 pb-16 bg-bg">
        <Suspense
          fallback={
            <div className="min-h-screen bg-bg flex flex-col items-center justify-center p-4">
              <RefreshCw className="animate-spin text-primary mb-4" size={32} />
              <p className="text-text-muted font-medium">Loading page modules...</p>
            </div>
          }
        >
          <ResultsContent />
        </Suspense>
      </main>
    </>
  );
}
