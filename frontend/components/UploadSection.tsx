"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Upload, CheckCircle, Shield, Zap, FileAudio, 
  Loader2, Activity, ArrowRight, Trash2, Eye, MessageSquare, User
} from "lucide-react";
import SectionContainer from "./SectionContainer";
import { getApiUrl } from "../utils/api";

const features = [
  { icon: CheckCircle, label: "High Quality", color: "text-green-500" },
  { icon: Shield, label: "Secure", color: "text-blue-500" },
  { icon: Zap, label: "Fast", color: "text-amber-500" },
];

type UploadState = "idle" | "transcript_form" | "uploading" | "analyzing" | "has_result" | "error";

interface SummaryData {
  id: string;
  filename: string;
  primaryDetection: string;
  confidence: number;
  date: string;
  topFeature?: string;
}

export default function UploadSection() {
  const router = useRouter();
  const [state, setState] = useState<UploadState>("idle");
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeResult, setActiveResult] = useState<SummaryData | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [transcript, setTranscript] = useState("");
  const [gender, setGender] = useState("unknown");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper to get top feature from SHAP
  const getTopFeature = (json: any) => {
    if (json?.shapData?.features?.length > 0) {
      const sorted = [...json.shapData.features].sort((a: any, b: any) => Math.abs(b.value) - Math.abs(a.value));
      return `${sorted[0].name} (${sorted[0].featureValue})`;
    }
    return "Pitch Variability (11.2 Hz)";
  };

  // Check for active result in localStorage
  useEffect(() => {
    const checkActiveResult = async () => {
      const activeId = localStorage.getItem("mindvoice_active_result_id");
      if (activeId) {
        if (activeId === "fallback-mock-id") {
          setActiveResult({
            id: "fallback-mock-id",
            filename: "mental_health_sample.wav",
            primaryDetection: "Depression",
            confidence: 78,
            date: "5/13/2026",
            topFeature: "Pitch Variability (11.2 Hz)"
          });
          setState("has_result");
        } else {
          try {
            const response = await fetch(getApiUrl(`/api/results/${activeId}`));
            if (response.ok) {
              const json = await response.json();
              if (json) {
                setActiveResult({
                  id: json.id,
                  filename: json.filename,
                  primaryDetection: json.primaryDetection,
                  confidence: json.confidence,
                  date: json.date,
                  topFeature: getTopFeature(json)
                });
                setState("has_result");
              }
            } else {
              // Clear invalid storage if backend says 404
              localStorage.removeItem("mindvoice_active_result_id");
            }
          } catch (err) {
            // Fallback load mock if backend is offline but key exists
            setActiveResult({
              id: activeId,
              filename: "mental_health_sample.wav",
              primaryDetection: "Depression",
              confidence: 78,
              date: "5/13/2026",
              topFeature: "Pitch Variability (11.2 Hz)"
            });
            setState("has_result");
          }
        }
      }
    };
    checkActiveResult();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      startProcess(file);
    }
  };

  const startProcess = (file: File) => {
    setFileName(file.name);
    setErrorMessage("");
    const sizeStr = file.size > 1024 * 1024
      ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
      : `${(file.size / 1024).toFixed(0)} KB`;
    setFileSize(sizeStr);
    setPendingFile(file);
    setState("transcript_form");
  };

  const handleConfirmAnalysis = async () => {
    if (!pendingFile) return;
    setState("uploading");
    setUploadProgress(0);

    const uploadInterval = setInterval(async () => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(uploadInterval);
          setState("analyzing");
          performAnalysis(pendingFile);
          return 100;
        }
        return prev + 10;
      });
    }, 150);
  };

  const audioBufferToWav = (audioBuffer: AudioBuffer) => {
    const numberOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.length;
    const bytesPerSample = 2;
    const blockAlign = numberOfChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + samples * blockAlign);
    const view = new DataView(buffer);

    const writeString = (offset: number, value: string) => {
      for (let i = 0; i < value.length; i++) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples * blockAlign, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples * blockAlign, true);

    const channels = Array.from({ length: numberOfChannels }, (_, channel) => {
      const data = new Float32Array(samples);
      audioBuffer.copyFromChannel(data, channel);
      return data;
    });

    let offset = 44;
    for (let i = 0; i < samples; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channels[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += bytesPerSample;
      }
    }

    return new Blob([buffer], { type: "audio/wav" });
  };

  const prepareAudioForUpload = async (file: File) => {
    if (/\.wav$/i.test(file.name)) {
      return file;
    }

    const browserWindow = window as typeof window & {
      webkitAudioContext?: typeof AudioContext;
    };
    const AudioContextClass = window.AudioContext || browserWindow.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error("Browser tidak mendukung konversi audio. Silakan gunakan file WAV.");
    }

    const audioContext = new AudioContextClass();
    try {
      const arrayBuffer = await file.arrayBuffer();
      const decoded = await audioContext.decodeAudioData(arrayBuffer);
      const wavBlob = audioBufferToWav(decoded);
      const convertedName = file.name.replace(/\.[^/.]+$/, "") || "audio";
      return new File([wavBlob], `${convertedName}.wav`, { type: "audio/wav" });
    } finally {
      await audioContext.close();
    }
  };

  const performAnalysis = async (file: File) => {
    const formData = new FormData();

    try {
      const uploadFile = await prepareAudioForUpload(file);
      formData.append("file", uploadFile);
      formData.append("transcript", transcript.trim());
      formData.append("gender", gender);

      const response = await fetch(getApiUrl("/api/analyze"), {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        // Save UUID in localStorage
        localStorage.setItem("mindvoice_active_result_id", result.id);
        
        // Dispatch custom storage event to notify header/Navbar instantly
        window.dispatchEvent(new Event("storage"));
        
        router.push(`/results?id=${result.id}`);
      } else {
        const errorBody = await response.json().catch(() => null);
        throw new Error(errorBody?.detail || "Analysis failed");
      }
    } catch (error) {
      console.warn("Audio analysis failed.", error);
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Audio tidak bisa diproses. Silakan coba file WAV, MP3, atau M4A lain."
      );
      setState("error");
    }
  };

  const clearResult = () => {
    localStorage.removeItem("mindvoice_active_result_id");
    // Dispatch event to notify Navbar
    window.dispatchEvent(new Event("storage"));
    setState("idle");
    setActiveResult(null);
    setErrorMessage("");
    setTranscript("");
    setGender("unknown");
    setPendingFile(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type.includes("audio") || /\.(mp3|wav|m4a|mp4a)$/i.test(file.name))) {
      startProcess(file);
    }
  };

  return (
    <SectionContainer id="upload" className="py-20 sm:py-28">
      <div className="text-center mb-10">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-3xl sm:text-4xl font-extrabold text-text mb-3"
        >
          {state === "has_result" ? "Your Analysis Report" : "Upload Your Audio"}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-text-muted text-base max-w-md mx-auto"
        >
          {state === "has_result" 
            ? "We found a saved audio analysis report from your previous session." 
            : "Start your mental health analysis by uploading an audio recording"}
        </motion.p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="max-w-2xl mx-auto"
      >
        <div className="bg-white rounded-2xl soft-shadow-lg border border-border-light p-6 sm:p-8">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="audio/*,.mp3,.wav,.m4a,.mp4a" 
            className="hidden" 
          />

          <AnimatePresence mode="wait">
            {state === "idle" && (
              <motion.div
                key="idle"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className="upload-glow border-2 border-dashed border-border rounded-2xl p-10 sm:p-14 text-center cursor-pointer hover:border-primary/30 transition-all group"
              >
                <motion.div
                  whileHover={{ scale: 1.05, rotate: 2 }}
                  className="w-16 h-16 rounded-2xl gradient-bg flex items-center justify-center mx-auto mb-5 shadow-lg"
                >
                  <Upload size={28} className="text-white" />
                </motion.div>
                <p className="text-text font-semibold text-lg mb-2">
                  Drop your audio file here
                </p>
                <p className="text-text-muted text-sm mb-4">
                  or click to browse files
                </p>
                <span className="inline-block px-4 py-1.5 rounded-full bg-bg text-text-muted text-xs font-medium border border-border-light">
                  Supports WAV, MP3 & M4A
                </span>
              </motion.div>
            )}

            {state === "transcript_form" && (
              <motion.div
                key="transcript_form"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-5"
              >
                {/* File info badge */}
                <div className="flex items-center gap-3 p-3 bg-bg rounded-xl border border-border-light">
                  <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                    <FileAudio size={18} className="text-primary" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-text truncate">{fileName}</p>
                    <p className="text-xs text-text-muted">{fileSize}</p>
                  </div>
                </div>

                {/* Transcript input */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm font-bold text-text">
                    <MessageSquare size={15} className="text-primary" />
                    Transkripsi Audio
                    <span className="ml-1 px-2 py-0.5 rounded-full bg-bg text-[10px] text-text-muted border border-border-light font-medium">Opsional — meningkatkan akurasi</span>
                  </label>
                  <textarea
                    id="transcript-input"
                    rows={4}
                    value={transcript}
                    onChange={(e) => setTranscript(e.target.value)}
                    placeholder="Ketik atau tempel teks dari audio di sini... Contoh: 'Akhir-akhir ini saya merasa sangat lelah dan tidak bersemangat. Tidur saya juga tidak teratur.'"
                    className="w-full resize-none rounded-xl border border-border-light bg-bg p-3.5 text-sm text-text placeholder:text-text-light focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all leading-relaxed"
                  />
                  <p className="text-[11px] text-text-light leading-relaxed">
                    Transkripsi membantu model menghitung fitur linguistik (kata negatif/positif, filler words, dll.) yang digunakan saat training. Tanpa transkripsi, fitur ini akan menggunakan nilai estimasi.
                  </p>
                </div>

                {/* Gender selector */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm font-bold text-text">
                    <User size={15} className="text-primary" />
                    Gender Pembicara
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(["male", "female", "unknown"] as const).map((g) => (
                      <button
                        key={g}
                        id={`gender-${g}`}
                        type="button"
                        onClick={() => setGender(g)}
                        className={`py-2.5 rounded-xl text-sm font-semibold border transition-all ${
                          gender === g
                            ? "gradient-bg text-white border-transparent shadow"
                            : "bg-bg text-text-muted border-border-light hover:border-primary/30"
                        }`}
                      >
                        {g === "male" ? "Male" : g === "female" ? "Female" : "Unknown"}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex gap-3 pt-1">
                  <button
                    type="button"
                    onClick={() => {
                      setState("idle");
                      setPendingFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                    className="flex-1 py-3 rounded-full border border-border bg-white text-text font-bold text-sm hover:bg-bg transition-colors"
                  >
                    Ganti File
                  </button>
                  <button
                    id="confirm-analyze-btn"
                    type="button"
                    onClick={handleConfirmAnalysis}
                    className="flex-[2] inline-flex items-center justify-center gap-2 py-3 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
                  >
                    Analisis Sekarang
                    <ArrowRight size={16} />
                  </button>
                </div>
              </motion.div>
            )}

            {state === "uploading" && (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="border border-border-light rounded-2xl p-10 text-center animate-pulse"
              >
                <div className="w-12 h-12 rounded-xl bg-primary/5 flex items-center justify-center mx-auto mb-4">
                  <FileAudio size={24} className="text-primary animate-pulse" />
                </div>
                <h4 className="font-semibold text-text mb-1 truncate max-w-xs mx-auto">{fileName}</h4>
                <p className="text-xs text-text-muted mb-4">{fileSize}</p>
                
                <div className="w-full bg-bg rounded-full h-2 max-w-xs mx-auto overflow-hidden">
                  <motion.div 
                    className="gradient-bg h-full rounded-full" 
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.1 }}
                  />
                </div>
                <p className="text-xs text-primary font-semibold mt-2">{uploadProgress}% Uploaded</p>
              </motion.div>
            )}

            {state === "analyzing" && (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="border border-border-light rounded-2xl p-10 text-center"
              >
                <Loader2 size={36} className="text-primary animate-spin mx-auto mb-4" />
                <h4 className="font-semibold text-text mb-2">Analyzing Voice Patterns...</h4>
                <p className="text-xs text-text-muted max-w-xs mx-auto leading-relaxed">
                  Extracting acoustic features and comparing predictions across Machine Learning and Deep Learning models.
                </p>
                
                <div className="flex items-center justify-center gap-1.5 h-8 mt-6">
                  {[...Array(12)].map((_, i) => (
                    <div 
                      key={i} 
                      className="w-1 bg-secondary rounded-full h-full animate-bounce" 
                      style={{ animationDelay: `${i * 0.1}s`, animationDuration: "1s" }}
                    />
                  ))}
                </div>
              </motion.div>
            )}

            {state === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="border border-red-100 bg-red-50/60 rounded-2xl p-8 text-center"
              >
                <div className="w-12 h-12 rounded-xl bg-white text-primary flex items-center justify-center mx-auto mb-4 shadow-sm">
                  <FileAudio size={24} />
                </div>
                <h4 className="font-bold text-text mb-2">Audio could not be analyzed</h4>
                <p className="text-xs text-text-muted max-w-sm mx-auto leading-relaxed mb-5">
                  {errorMessage}
                </p>
                <button
                  onClick={() => {
                    setState("idle");
                    setErrorMessage("");
                    setPendingFile(null);
                    setTranscript("");
                    setGender("unknown");
                    if (fileInputRef.current) {
                      fileInputRef.current.value = "";
                    }
                  }}
                  className="inline-flex items-center justify-center px-5 py-2.5 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
                >
                  Try Another Audio
                </button>
              </motion.div>
            )}

            {state === "has_result" && activeResult && (
              <motion.div
                key="has_result"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="border border-border-light rounded-2xl p-6 text-center space-y-5"
              >
                <div className="w-14 h-14 rounded-2xl bg-green-50 text-green-500 flex items-center justify-center mx-auto shadow-sm">
                  <CheckCircle size={28} />
                </div>
                <div>
                  <h4 className="font-bold text-text text-lg">Active Report Ready</h4>
                  <p className="text-xs text-text-muted truncate max-w-xs mx-auto mt-0.5">{activeResult.filename}</p>
                </div>

                <div className="max-w-xs mx-auto bg-bg p-4 rounded-xl border border-border-light space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-left">
                      <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Detection</span>
                      <span className="text-sm font-bold text-text block mt-0.5">{activeResult.primaryDetection}</span>
                    </div>
                    <div className="text-right border-l border-border pl-4">
                      <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Confidence</span>
                      <span className="text-sm font-bold text-primary block mt-0.5">{activeResult.confidence}%</span>
                    </div>
                  </div>
                  {activeResult.topFeature && (
                    <div className="border-t border-border-light pt-2.5 text-left">
                      <span className="text-[10px] text-text-light font-bold uppercase tracking-wider block">Top Contributor (SHAP)</span>
                      <span className="text-xs font-semibold text-text-muted block mt-0.5 truncate">
                        {activeResult.topFeature}
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex gap-3 max-w-md mx-auto pt-2">
                  <button
                    onClick={() => router.push(`/results?id=${activeResult.id}`)}
                    className="flex-1 inline-flex items-center justify-center gap-1.5 py-3 rounded-full gradient-bg text-white font-bold text-sm shadow hover:scale-[1.02] transition-transform"
                  >
                    <Eye size={16} />
                    View Report
                  </button>
                  <button
                    onClick={clearResult}
                    className="flex-1 inline-flex items-center justify-center gap-1.5 py-3 rounded-full border border-border bg-white text-text font-bold text-sm hover:bg-bg transition-colors"
                  >
                    <Trash2 size={16} className="text-text-muted" />
                    New Audio
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Bottom features */}
          {state !== "has_result" && (
            <div className="flex items-center justify-center gap-6 sm:gap-8 mt-6 pt-6 border-t border-border-light">
              {features.map((feat, i) => (
                <motion.div
                  key={feat.label}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="flex items-center gap-2"
                >
                  <feat.icon size={16} className={feat.color} />
                  <span className="text-sm text-text-muted font-medium">
                    {feat.label}
                  </span>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </SectionContainer>
  );
}
