"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { 
  Activity, Play, Pause, Brain, Clock, AlertCircle, 
  RefreshCw, Info 
} from "lucide-react";
import Link from "next/link";
import { getApiUrl } from "@/utils/api";

type AudioResultData = {
  id: string;
  audioInfo?: {
    audioUrl?: string;
  };
};

const resolveAudioUrl = (data: AudioResultData) => {
  const rawAudioUrl = data.audioInfo?.audioUrl || `/api/audio/${data.id}`;
  return rawAudioUrl.startsWith("http") ? rawAudioUrl : getApiUrl(rawAudioUrl);
};

function ResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const resultId = searchParams.get("id");

  const [data, setData] = useState<any>(null);
  const [resultError, setResultError] = useState("");
  const [loading, setLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0.0);
  const [totalDuration, setTotalDuration] = useState(45.0);
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);
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
        const json = await response.json();
        setData(json);
        if (json.audioInfo?.duration) {
          const parsed = parseFloat(json.audioInfo.duration);
          if (!isNaN(parsed)) {
            setTotalDuration(parsed);
          }
        }
      } catch (err) {
        console.warn("Backend fetch failed.", err);
        setResultError("Analysis result could not be loaded. Please upload and analyze the audio again.");
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [resultId]);

  // Decode audio to extract waveform peaks dynamically
  useEffect(() => {
    if (loading || !data) return;

    const audioUrl = resolveAudioUrl(data);
    
    if (data.id === "fallback-mock-id") {
      // Mock static peaks
      const peaks = Array.from({ length: 64 }, (_, idx) => {
        return Math.abs(Math.sin(idx * 0.15)) * 0.8 + 0.1;
      });
      setWaveformPeaks(peaks);
      return;
    }

    const loadWaveform = async () => {
      try {
        const response = await fetch(audioUrl);
        if (!response.ok) throw new Error("Failed to fetch audio file");
        const arrayBuffer = await response.arrayBuffer();
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const audioCtx = new AudioContextClass();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        
        // Extract amplitude peaks
        const rawData = audioBuffer.getChannelData(0);
        const samples = 64;
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];
        for (let i = 0; i < samples; i++) {
          let blockStart = blockSize * i;
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[blockStart + j]);
          }
          filteredData.push(sum / blockSize);
        }
        
        // Normalize
        const max = Math.max(...filteredData) || 1;
        const normalized = filteredData.map(val => (val / max) * 0.8 + 0.15);
        setWaveformPeaks(normalized);
      } catch (err) {
        console.warn("Could not decode audio data for waveform, generating seed-based peaks", err);
        const peaks = Array.from({ length: 64 }, (_, idx) => {
          const seed = data.id.split("").reduce((acc: number, char: string) => acc + char.charCodeAt(0), 0);
          return Math.abs(Math.sin(idx * 0.1 + seed)) * 0.75 + 0.15;
        });
        setWaveformPeaks(peaks);
      }
    };
    
    loadWaveform();
  }, [data, loading]);

  const handlePlayToggle = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch(err => {
        console.error("Audio play failed", err);
      });
    }
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
        <h2 className="text-xl font-extrabold text-text mb-2">Analysis Result Unavailable</h2>
        <p className="text-sm text-text-muted max-w-md leading-relaxed mb-6">
          {resultError || "The analysis result could not be loaded. Please upload the audio again."}
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

  // Circular progress properties for perfect circle
  const radius = 58;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (data.confidence / 100) * circumference;

  // Active color based on detection
  const isDepression = data.primaryDetection === "Depression";
  const activeColor = isDepression
    ? { stroke: "#E91E63", text: "text-primary", bg: "bg-red-500" }
    : { stroke: "#10B981", text: "text-green-500", bg: "bg-green-500" };

  // AI Interpretation dynamic text selection based on depression percentage
  const depressionScore = Math.max(0, Math.min(100, Number(data.metrics?.depression ?? 0)));
  let recommendationTitle = "";
  let recommendationText = "";

  if (depressionScore <= 20) {
    recommendationTitle = "Very Low Depression Indicators";
    recommendationText = `Depression indicator: ${depressionScore}%. The uploaded audio shows vocal patterns that are mostly aligned with a stable emotional state. The model detects low risk markers in this recording.

Recommendations:
1. Maintain your current healthy routine, including sleep, hydration, and balanced daily activity.
2. Keep doing regular self-check-ins, such as journaling once or twice a week.
3. Use this result as a baseline for future comparisons, especially if your mood or stress level changes.`;
  } else if (depressionScore <= 40) {
    recommendationTitle = "Low to Mild Emotional Strain";
    recommendationText = `Depression indicator: ${depressionScore}%. The audio contains a few markers that may reflect mild stress, fatigue, or temporary emotional strain, but the overall pattern is still closer to a normal state.

Recommendations:
1. Take short breaks during the day and reduce avoidable sources of stress where possible.
2. Prioritize consistent sleep and light physical activity for the next few days.
3. Recheck with another recording if you feel your mood, energy, or motivation is declining.`;
  } else if (depressionScore <= 60) {
    recommendationTitle = "Moderate Emotional Fluctuation";
    recommendationText = `Depression indicator: ${depressionScore}%. The model finds a balanced mix of normal and depression-related vocal markers. This may suggest emotional fluctuation, stress accumulation, or reduced vocal energy.

Recommendations:
1. Monitor your mood and daily functioning more intentionally over the next week.
2. Talk with a trusted friend, family member, mentor, or counselor if the feeling persists.
3. Try structured coping activities such as breathing exercises, a short walk, or breaking tasks into smaller steps.`;
  } else if (depressionScore <= 80) {
    recommendationTitle = "High Depression-Related Voice Markers";
    recommendationText = `Depression indicator: ${depressionScore}%. The uploaded audio shows stronger vocal markers associated with depression, such as reduced variation, lower energy, or slower speech-related patterns.

Recommendations:
1. Consider reaching out to a mental health professional, campus counselor, or trusted support person.
2. Avoid handling this alone if the symptoms affect sleep, appetite, motivation, study, or work.
3. Create a simple support plan today: one person to contact, one small task to complete, and one calming activity.`;
  } else {
    recommendationTitle = "Very High Depression-Related Indicators";
    recommendationText = `Depression indicator: ${depressionScore}%. The model detects very strong depression-related vocal markers in this recording. This result should be treated as an important signal for follow-up, not as a clinical diagnosis.

Recommendations:
1. Please seek support from a qualified mental health professional as soon as possible.
2. If you feel unsafe, overwhelmed, or at risk of self-harm, contact local emergency services or a crisis hotline immediately.
3. Reach out to someone you trust today and avoid staying isolated while waiting for professional help.`;
  }

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10 space-y-6 sm:space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-text-muted mb-2">
          <span>Analysis Complete</span>
          <span>•</span>
          <span>{data.date}</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-text mb-1">Analysis Results</h2>
        <p className="text-text-muted text-sm sm:text-base">
          Comprehensive mental health pattern detection from audio analysis
        </p>
      </div>

      {/* Primary Detection Card */}
      <div className="bg-white rounded-2xl soft-shadow border border-border-light p-5 sm:p-7 grid md:grid-cols-3 gap-6 sm:gap-8 items-center">
        <div className="md:col-span-2 space-y-4">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-50 text-xs font-semibold text-primary border border-red-100/50">
            <Activity size={12} />
            Primary Detection
          </div>
          <div>
            <h3 className="text-2xl sm:text-3xl font-black text-text tracking-tight mb-2">
              {data.primaryDetection}
            </h3>
            <p className="text-sm text-text-muted leading-relaxed">
              The audio pattern indicates voice features associated with {isDepression ? "depression" : "a healthy / normal state"}. 
              Prosodic parameters are key contributors to this consensus prediction.
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
          </div>

          <div className="flex items-start gap-2.5 p-3.5 bg-red-50/50 border border-red-100/50 rounded-xl">
            <Info size={16} className="text-primary shrink-0 mt-0.5" />
            <p className="text-[11px] sm:text-xs text-text-muted leading-relaxed">
              <strong className="text-text">Research Context:</strong> This analysis is for research purposes only and should not replace professional medical diagnosis.
            </p>
          </div>
        </div>

        {/* Circular Progress Gauge - Perfectly Circular Shape */}
        <div className="flex flex-col items-center justify-center p-4">
          <div className="relative flex items-center justify-center w-[140px] h-[140px]">
            <svg
              viewBox="0 0 140 140"
              className="w-full h-full transform -rotate-90 select-none"
            >
              {/* Background track */}
              <circle
                stroke="#F1F5F9"
                fill="transparent"
                strokeWidth="12"
                r={radius}
                cx="70"
                cy="70"
              />
              {/* Foreground stroke */}
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
              <p className="text-3xl font-black text-text leading-none">{data.confidence}%</p>
              <p className="text-[10px] text-text-light font-bold uppercase tracking-wider mt-1">Confidence</p>
            </div>
          </div>
        </div>
      </div>

      {/* Grid of 2 Small Cards (Depression & Normal only) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Depression */}
        <div className="bg-white rounded-xl border border-border-light p-4 soft-shadow">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-bold text-text-muted">Depression</span>
            <Activity size={14} className="text-primary" />
          </div>
          <h4 className="text-xl sm:text-2xl font-black text-text mb-2">{data.metrics.depression}%</h4>
          <div className="w-full bg-bg h-1.5 rounded-full overflow-hidden">
            <div className="bg-primary h-full rounded-full" style={{ width: `${data.metrics.depression}%` }} />
          </div>
        </div>

        {/* Normal */}
        <div className="bg-white rounded-xl border border-border-light p-4 soft-shadow">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-bold text-text-muted">Normal</span>
            <Activity size={14} className="text-green-500" />
          </div>
          <h4 className="text-xl sm:text-2xl font-black text-text mb-2">{data.metrics.normal}%</h4>
          <div className="w-full bg-bg h-1.5 rounded-full overflow-hidden">
            <div className="bg-green-500 h-full rounded-full" style={{ width: `${data.metrics.normal}%` }} />
          </div>
        </div>
      </div>

      {/* Audio Info Card Grid */}
      {data.audioInfo && (
        <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
              <Clock size={18} />
            </div>
            <div>
              <h4 className="text-sm sm:text-base font-extrabold text-text">Acoustic Biomarkers & Audio Details</h4>
              <p className="text-xs text-text-muted">Acoustic properties extracted from the analyzed speech signal</p>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-bg p-3.5 rounded-xl border border-border-light">
              <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Duration</span>
              <span className="text-sm font-extrabold text-text block mt-0.5">{data.audioInfo.duration}</span>
            </div>
            <div className="bg-bg p-3.5 rounded-xl border border-border-light">
              <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Average Pitch</span>
              <span className="text-sm font-extrabold text-text block mt-0.5">{data.audioInfo.avgPitch}</span>
            </div>
            <div className="bg-bg p-3.5 rounded-xl border border-border-light">
              <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Vocal Energy</span>
              <span className="text-sm font-extrabold text-text block mt-0.5">{data.audioInfo.energyLevel}</span>
            </div>
            <div className="bg-bg p-3.5 rounded-xl border border-border-light">
              <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Signal Quality</span>
              <span className="text-sm font-extrabold text-text block mt-0.5">{data.audioInfo.signalQuality}</span>
            </div>
          </div>
        </div>
      )}

      {/* AI Interpretation (With Dynamic Recommendation Template) */}
      <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-6">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
            <Brain size={20} />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-extrabold text-text">AI Interpretation & Recommendations</h4>
            <p className="text-xs text-text-muted">{recommendationTitle}</p>
          </div>
        </div>

        <p className="text-xs sm:text-sm text-text-muted leading-relaxed whitespace-pre-line bg-bg p-4 rounded-xl border border-border-light">
          {recommendationText}
        </p>

        {/* Model Performance metrics */}
        <div className="pt-4 border-t border-border-light">
          <p className="text-[10px] text-text-light font-bold uppercase tracking-wider mb-3">Model Performance Metrics:</p>
          <div className="grid grid-cols-3 gap-4 bg-bg rounded-xl p-3.5 text-center">
            <div>
              <p className="text-xs font-bold text-text-muted mb-0.5">Accuracy</p>
              <p className="text-base sm:text-lg font-black text-text">{data.performance.accuracy}</p>
            </div>
            <div className="border-x border-border">
              <p className="text-xs font-bold text-text-muted mb-0.5">Precision</p>
              <p className="text-base sm:text-lg font-black text-text">{data.performance.precision}</p>
            </div>
            <div>
              <p className="text-xs font-bold text-text-muted mb-0.5">F1-Score</p>
              <p className="text-base sm:text-lg font-black text-text">{data.performance.f1Score}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Audio Waveform Controller (Real audio playback & dynamic visualization) */}
      <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-6 soft-shadow space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-border-light">
          <h4 className="text-sm sm:text-base font-extrabold text-text">Audio Waveform</h4>
          <span className="text-xs font-medium text-text-muted">{data.filename}</span>
        </div>

        {/* Hidden Audio element */}
        <audio 
          ref={audioRef}
          src={resolveAudioUrl(data)}
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
          onEnded={() => {
            setIsPlaying(false);
            setCurrentTime(0);
          }}
        />

        {/* Waveform Visualization */}
        <div className="bg-bg rounded-xl p-4 flex items-end justify-between h-20 gap-[2px] select-none">
          {(waveformPeaks.length > 0 ? waveformPeaks : Array.from({ length: 64 }).map(() => 0.15)).map((peak, idx) => {
            const h = peak * 90; // scale to max 90%
            // Calculate if the current bar has been "played"
            const isPlayed = idx / 64 < playProgress;
            return (
              <div
                key={idx}
                className={`w-[3px] rounded-full transition-all duration-100 ${
                  isPlayed ? "gradient-bg opacity-90" : "bg-slate-200"
                }`}
                style={{ height: `${h}%` }}
              />
            );
          })}
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-4">
          <button 
            onClick={handlePlayToggle}
            className="w-10 h-10 rounded-full gradient-bg flex items-center justify-center text-white shadow hover:scale-105 transition-transform"
          >
            {isPlaying ? <Pause size={18} fill="white" /> : <Play size={18} fill="white" className="ml-0.5" />}
          </button>
          
          <div className="flex-1 relative flex items-center">
            <input 
              type="range"
              min="0"
              max={totalDuration}
              step="0.05"
              value={currentTime}
              onChange={handleSliderChange}
              className="w-full accent-primary h-1 rounded-lg cursor-pointer bg-slate-100"
            />
          </div>

          <span className="text-xs font-bold text-text-muted shrink-0 w-20 text-right">
            {formatTime(currentTime)} / {formatTime(totalDuration)}
          </span>
        </div>
      </div>

      {/* Navigation Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 pt-4">
        <Link 
          href={(resultId || data?.id) ? `/ai-insight?id=${resultId || data?.id}` : "/ai-insight"}
          className="flex-1 text-center py-3.5 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
        >
          View AI Insights
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
        <Suspense fallback={
          <div className="min-h-screen bg-bg flex flex-col items-center justify-center p-4">
            <RefreshCw className="animate-spin text-primary mb-4" size={32} />
            <p className="text-text-muted font-medium">Loading page modules...</p>
          </div>
        }>
          <ResultsContent />
        </Suspense>
      </main>
    </>
  );
}
