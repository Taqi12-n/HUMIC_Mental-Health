"use client";

import { motion } from "framer-motion";
import { Upload, Cpu, BarChart3 } from "lucide-react";
import SectionContainer from "./SectionContainer";

const steps = [
  {
    number: 1,
    icon: Upload,
    title: "Upload Audio",
    description:
      "Upload your voice recording in WAV or MP3 format. Our system accepts recordings of any length.",
  },
  {
    number: 2,
    icon: Cpu,
    title: "AI Analysis",
    description:
      "Advanced ML and DL models analyze vocal patterns, tone variations, and acoustic features in real-time.",
  },
  {
    number: 3,
    icon: BarChart3,
    title: "Get Insights",
    description:
      "Receive detailed mental health insights with confidence scores, visualizations, and actionable recommendations.",
  },
];

export default function HowItWorks() {
  return (
    <SectionContainer id="how-it-works" className="py-20 sm:py-28">
      <div className="text-center mb-14">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-3xl sm:text-4xl font-extrabold text-text mb-3"
        >
          How It Works
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-text-muted text-base max-w-md mx-auto"
        >
          Three simple steps to analyze your mental health through voice
        </motion.p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
        {steps.map((step, i) => (
          <motion.div
            key={step.number}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.15 }}
            whileHover={{ y: -6, transition: { duration: 0.25 } }}
            className="relative bg-white rounded-2xl soft-shadow p-7 border border-border-light hover:soft-shadow-lg transition-shadow group"
          >
            {/* Number badge */}
            <div className="absolute -top-3 -right-3 number-badge shadow-md">
              {step.number}
            </div>

            {/* Icon */}
            <div className="w-12 h-12 rounded-xl gradient-bg-subtle flex items-center justify-center mb-5 group-hover:scale-105 transition-transform">
              <step.icon size={22} className="text-primary" />
            </div>

            <h3 className="text-lg font-bold text-text mb-2">{step.title}</h3>
            <p className="text-text-muted text-sm leading-relaxed">
              {step.description}
            </p>
          </motion.div>
        ))}
      </div>
    </SectionContainer>
  );
}
