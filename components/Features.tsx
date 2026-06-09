"use client";

import { motion } from "framer-motion";
import { AudioLines, GitCompareArrows, Eye, TrendingUp } from "lucide-react";
import SectionContainer from "./SectionContainer";

const features = [
  {
    icon: AudioLines,
    title: "Audio-Based Detection",
    description:
      "Analyze vocal biomarkers and speech patterns to identify mental health indicators with high precision.",
  },
  {
    icon: GitCompareArrows,
    title: "ML vs DL Comparison",
    description:
      "Compare machine learning and deep learning model results side by side for comprehensive analysis.",
  },
  {
    icon: Eye,
    title: "XAI Visualization",
    description:
      "Explainable AI visualizations help understand which audio features drive the model predictions.",
  },
  {
    icon: TrendingUp,
    title: "Confidence Analytics",
    description:
      "Detailed confidence scores and analytics dashboard for each prediction across multiple models.",
  },
];

export default function Features() {
  return (
    <SectionContainer id="features" className="py-20 sm:py-28">
      <div className="text-center mb-14">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-3xl sm:text-4xl font-extrabold text-text mb-3"
        >
          Feature Highlights
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-text-muted text-base max-w-md mx-auto"
        >
          Powerful tools designed for accurate mental health voice analysis
        </motion.p>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {features.map((feature, i) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            whileHover={{ y: -4, transition: { duration: 0.25 } }}
            className="bg-white rounded-2xl soft-shadow p-6 border border-border-light hover:soft-shadow-lg transition-shadow group"
          >
            <div className="w-11 h-11 rounded-xl gradient-bg-subtle flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
              <feature.icon size={20} className="text-primary" />
            </div>
            <h3 className="text-base font-bold text-text mb-2">
              {feature.title}
            </h3>
            <p className="text-text-muted text-sm leading-relaxed">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </div>
    </SectionContainer>
  );
}
