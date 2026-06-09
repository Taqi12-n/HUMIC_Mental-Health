"use client";

import { motion } from "framer-motion";
import { ArrowRight, Play, Activity, Cpu, BarChart3 } from "lucide-react";

function WaveformVisualization() {
  const heights = [
    15, 25, 45, 60, 35, 20, 30, 55, 75, 50, 40, 65, 85, 95, 80, 50, 
    30, 45, 70, 85, 65, 35, 20, 40, 60, 75, 50, 30, 20, 15, 25, 15
  ];
  return (
    <div className="flex items-end justify-center gap-[3px] h-16 px-2">
      {heights.map((height, i) => (
        <div
          key={i}
          className="w-[4px] rounded-full gradient-bg opacity-70"
          style={{
            height: `${height}%`,
            minHeight: "4px",
          }}
        />
      ))}
    </div>
  );
}

export default function Hero() {
  return (
    <section
      id="home"
      className="relative pt-28 sm:pt-36 pb-16 sm:pb-24 overflow-hidden"
    >
      {/* Background decorations */}
      <div className="absolute top-20 right-0 w-[500px] h-[500px] bg-gradient-to-br from-primary/5 to-secondary/5 rounded-full blur-3xl -z-10" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-gradient-to-tr from-secondary/5 to-primary/5 rounded-full blur-3xl -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full gradient-bg-subtle border border-primary/10 mb-6"
            >
              <Activity size={14} className="text-primary" />
              <span className="text-xs font-semibold text-primary">
                Research-Grade AI Analysis
              </span>
            </motion.div>

            <h1 className="text-4xl sm:text-5xl lg:text-[3.5rem] font-extrabold text-text leading-[1.1] tracking-tight mb-6">
              Detect Mental Health
              <br />
              Patterns Through{" "}
              <span className="gradient-text">Voice</span>
            </h1>

            <p className="text-base sm:text-lg text-text-muted leading-relaxed mb-8 max-w-lg">
              Advanced AI-powered voice analysis to detect mental health patterns
              with research-grade accuracy. Upload audio and get instant
              insights.
            </p>

            <div className="flex flex-wrap gap-3">
              <motion.a
                href="#upload"
                whileHover={{ scale: 1.03, boxShadow: "0 8px 30px rgba(233,30,99,0.25)" }}
                whileTap={{ scale: 0.98 }}
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full gradient-bg text-white font-semibold text-sm shadow-lg transition-all"
              >
                <Play size={16} fill="white" />
                Analyze Audio
              </motion.a>
              <motion.a
                href="#how-it-works"
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full border border-border text-text font-semibold text-sm hover:border-primary/30 hover:bg-primary/[0.02] transition-all"
              >
                Learn More
                <ArrowRight size={16} />
              </motion.a>
            </div>
          </motion.div>

          {/* Right analytics card */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3, ease: "easeOut" }}
            className="animate-float"
          >
            <div className="bg-white rounded-2xl soft-shadow-lg p-6 border border-border-light">
              {/* Card Header */}
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg gradient-bg flex items-center justify-center">
                    <Activity size={16} className="text-white" />
                  </div>
                  <span className="font-semibold text-text text-sm">
                    Audio Analysis
                  </span>
                </div>
                <span className="px-3 py-1 rounded-full bg-green-50 text-green-600 text-xs font-semibold border border-green-100">
                  Active
                </span>
              </div>

              {/* Waveform */}
              <div className="bg-bg rounded-xl p-4 mb-5">
                <WaveformVisualization />
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-bg rounded-xl p-3 text-center">
                  <div className="flex items-center justify-center mb-1">
                    <BarChart3 size={14} className="text-primary" />
                  </div>
                  <p className="text-lg font-bold text-text">99.9%</p>
                  <p className="text-[11px] text-text-muted">Accuracy</p>
                </div>
                <div className="bg-bg rounded-xl p-3 text-center">
                  <div className="flex items-center justify-center mb-1">
                    <Activity size={14} className="text-secondary" />
                  </div>
                  <p className="text-lg font-bold text-text">1,247</p>
                  <p className="text-[11px] text-text-muted">Processed</p>
                </div>
                <div className="bg-bg rounded-xl p-3 text-center">
                  <div className="flex items-center justify-center mb-1">
                    <Cpu size={14} className="text-primary" />
                  </div>
                  <p className="text-lg font-bold text-text">2</p>
                  <p className="text-[11px] text-text-muted">Models</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
