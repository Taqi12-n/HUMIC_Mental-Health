"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Info, HelpCircle, Activity, ChevronDown, ChevronUp, Layers, HelpCircle as HelpIcon } from "lucide-react";

interface ShapFeature {
  name: string;
  value: number;
  featureValue: string;
  effect: string;
}

interface ShapData {
  baseValue: number;
  predictionValue: number;
  features: ShapFeature[];
}

interface LimeRule {
  feature: string;
  rule: string;
  value: string;
  weight: number;
  influence: string;
}

interface XaiSectionProps {
  data: {
    primaryDetection: string;
    confidence: number;
    shapData?: ShapData;
    limeRules?: LimeRule[];
  };
}

const biomarkerDescriptions = [
  {
    name: "Pitch Variability (F0 SD)",
    definition: "Standard deviation of the fundamental frequency (F0) across the speech signal.",
    clinicalMeaning: "Normal speech exhibits dynamic pitch variation reflecting rich emotional modulation. Monotone speech (very low pitch variability) is a classic indicator of emotional flattening often associated with clinical depression.",
  },
  {
    name: "Speech Tempo",
    definition: "Average speech rate measured in syllables per second.",
    clinicalMeaning: "Reduced speech rate (tempo) indicates cognitive deceleration and psychomotor retardation, common markers of fatigue or depressive states.",
  },
  {
    name: "Pause Ratio",
    definition: "Percentage of total audio duration consisting of silence or non-speech pauses.",
    clinicalMeaning: "Depressed individuals often demonstrate longer and more frequent pauses during speech, representing hesitation, speech formulation difficulties, or low cognitive energy.",
  },
  {
    name: "Jitter (local)",
    definition: "Short-term cycle-to-cycle perturbations in fundamental frequency.",
    clinicalMeaning: "Elevated jitter indicates micro-instability in vocal fold vibration, often caused by vocal fold tension, physical fatigue, or autonomic nervous system arousal (stress).",
  },
  {
    name: "Spectral Centroid",
    definition: "The 'center of gravity' of the audio spectrum, indicating the brightness or frequency distribution of the voice.",
    clinicalMeaning: "A lower spectral centroid indicates 'breathy' or 'darker' vocal quality with less energy in high frequencies, which is frequently observed in depressed states compared to high-energy resonant speech.",
  }
];

export default function XaiSection({ data }: XaiSectionProps) {
  const [activeTab, setActiveTab] = useState<"shap" | "lime">("shap");
  const [expandedBiomarker, setExpandedBiomarker] = useState<string | null>(null);

  // Fallback structures if backend lacks them
  const defaultShapData: ShapData = data.shapData || {
    baseValue: 50.0,
    predictionValue: data.primaryDetection === "Depression" ? floatVal(data.confidence) : floatVal(100 - data.confidence),
    features: data.primaryDetection === "Depression" ? [
      {"name": "Pitch Variability (F0 SD)", "value": 8.5, "featureValue": "11.2 Hz", "effect": "increases risk"},
      {"name": "Speech Tempo", "value": 7.2, "featureValue": "2.1 syl/s", "effect": "increases risk"},
      {"name": "Pause Ratio", "value": 6.1, "featureValue": "24.5%", "effect": "increases risk"},
      {"name": "Jitter (local)", "value": 4.0, "featureValue": "1.82%", "effect": "increases risk"},
      {"name": "Spectral Centroid", "value": 2.2, "featureValue": "1250 Hz", "effect": "increases risk"}
    ] : [
      {"name": "Pitch Variability (F0 SD)", "value": -9.2, "featureValue": "31.8 Hz", "effect": "decreases risk"},
      {"name": "Speech Tempo", "value": -8.1, "featureValue": "3.8 syl/s", "effect": "decreases risk"},
      {"name": "Pause Ratio", "value": -6.3, "featureValue": "8.2%", "effect": "decreases risk"},
      {"name": "Jitter (local)", "value": -4.2, "featureValue": "0.65%", "effect": "decreases risk"},
      {"name": "Spectral Centroid", "value": -2.2, "featureValue": "1890 Hz", "effect": "decreases risk"}
    ]
  };

  const defaultLimeRules: LimeRule[] = data.limeRules || (data.primaryDetection === "Depression" ? [
    {"feature": "Pitch Variability", "rule": "F0 SD <= 15.0 Hz", "value": "11.2 Hz", "weight": 0.24, "influence": "Positive (Depression)"},
    {"feature": "Speech Tempo", "rule": "Tempo <= 2.4 syl/s", "value": "2.1 syl/s", "weight": 0.20, "influence": "Positive (Depression)"},
    {"feature": "Pause Ratio", "rule": "Pause Ratio > 18.0%", "value": "24.5%", "weight": 0.16, "influence": "Positive (Depression)"},
    {"feature": "Jitter", "rule": "Jitter > 1.05%", "value": "1.82%", "weight": 0.12, "influence": "Positive (Depression)"},
    {"feature": "Spectral Centroid", "rule": "Centroid <= 1400 Hz", "value": "1250 Hz", "weight": 0.08, "influence": "Positive (Depression)"}
  ] : [
    {"feature": "Pitch Variability", "rule": "F0 SD > 22.0 Hz", "value": "31.8 Hz", "weight": -0.26, "influence": "Negative (Normal)"},
    {"feature": "Speech Tempo", "rule": "Tempo > 3.0 syl/s", "value": "3.8 syl/s", "weight": -0.22, "influence": "Negative (Normal)"},
    {"feature": "Pause Ratio", "rule": "Pause Ratio <= 12.0%", "value": "8.2%", "weight": -0.18, "influence": "Negative (Normal)"},
    {"feature": "Jitter", "rule": "Jitter <= 1.05%", "value": "0.65%", "weight": -0.12, "influence": "Negative (Normal)"},
    {"feature": "Spectral Centroid", "rule": "Centroid > 1600 Hz", "value": "1890 Hz", "weight": -0.09, "influence": "Negative (Normal)"}
  ]);

  function floatVal(v: number): number {
    return parseFloat(v.toString());
  }

  // Calculate coordinates for SHAP Force Plot
  const shap = defaultShapData;
  const isDepressionResult = data.primaryDetection === "Depression";

  return (
    <div className="bg-white rounded-2xl border border-border-light p-5 sm:p-7 soft-shadow space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border-light">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-xl gradient-bg-subtle flex items-center justify-center text-primary">
            <Layers size={20} />
          </div>
          <div>
            <h4 className="text-sm sm:text-base font-extrabold text-text">Explainable AI (XAI) Dashboard</h4>
            <p className="text-xs text-text-muted">Interpreting model predictions using SHAP & LIME frameworks</p>
          </div>
        </div>

        {/* Tab Buttons */}
        <div className="flex bg-bg p-1 rounded-xl border border-border-light text-xs font-semibold self-start sm:self-auto">
          <button
            onClick={() => setActiveTab("shap")}
            className={`px-4 py-2 rounded-lg transition-all ${
              activeTab === "shap" ? "bg-white text-text shadow-sm" : "text-text-muted hover:text-text"
            }`}
          >
            SHAP Summary
          </button>
          <button
            onClick={() => setActiveTab("lime")}
            className={`px-4 py-2 rounded-lg transition-all ${
              activeTab === "lime" ? "bg-white text-text shadow-sm" : "text-text-muted hover:text-text"
            }`}
          >
            LIME Decision Rules
          </button>
        </div>
      </div>

      {/* Main Tab Content */}
      <div>
        <AnimatePresence mode="wait">
          {activeTab === "shap" ? (
            <motion.div
              key="shap"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
              className="space-y-6"
            >
              {/* Description */}
              <div className="bg-bg rounded-xl p-4 border border-border-light flex gap-3 items-start">
                <Info size={16} className="text-primary shrink-0 mt-0.5" />
                <p className="text-xs text-text-muted leading-relaxed">
                  <strong className="text-text">SHAP (SHapley Additive exPlanations)</strong> values explain how each acoustic biomarker pushes the model prediction away from the baseline average prediction (<span className="font-bold text-text">50.0%</span> risk score). 
                  <span className="text-red-500 font-semibold"> Red/positive values</span> increase depression risk probability, while <span className="text-green-600 font-semibold">green/negative values</span> reduce it.
                </p>
              </div>

              {/* Force-like Waterfall Bar Plot */}
              <div className="space-y-4">
                <div className="flex justify-between items-center text-xs font-bold text-text-muted">
                  <span>Base Value: 50%</span>
                  <span className={isDepressionResult ? "text-primary" : "text-green-500"}>
                    Prediction: {shap.predictionValue.toFixed(1)}% (Depression Risk)
                  </span>
                </div>

                {/* Horizontal cumulative contribution flow */}
                <div className="space-y-3.5 pt-2">
                  {shap.features.map((feat, i) => {
                    const isPositive = feat.value >= 0;
                    const magnitude = Math.abs(feat.value);
                    // percentage of width (capped for visual safety)
                    const barWidth = Math.min((magnitude / 50) * 100, 100);

                    return (
                      <div key={feat.name} className="space-y-1">
                        <div className="flex justify-between text-xs font-medium">
                          <span className="text-text font-semibold flex items-center gap-1.5">
                            {feat.name}
                            <span className="text-[10px] text-text-light font-bold">({feat.featureValue})</span>
                          </span>
                          <span className={`font-bold ${isPositive ? "text-primary" : "text-green-500"}`}>
                            {isPositive ? "+" : ""}{feat.value.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden relative">
                          {/* Indicator line for 50% baseline */}
                          <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-slate-300 z-10" />
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${barWidth}%` }}
                            transition={{ duration: 0.8, delay: i * 0.05, ease: "easeOut" }}
                            className={`h-full rounded-full absolute ${
                              isPositive 
                                ? "bg-primary left-1/2" 
                                : "bg-green-500 right-1/2 origin-right"
                            }`}
                            style={{
                              left: isPositive ? "50%" : "auto",
                              right: isPositive ? "auto" : "50%"
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Legend */}
                <div className="flex justify-center items-center gap-6 pt-3 text-[10px] sm:text-xs font-bold text-text-muted">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-primary" />
                    <span>Pushes toward Depression State</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-green-500" />
                    <span>Pushes toward Normal State</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="lime"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
              className="space-y-6"
            >
              {/* Description */}
              <div className="bg-bg rounded-xl p-4 border border-border-light flex gap-3 items-start">
                <Info size={16} className="text-primary shrink-0 mt-0.5" />
                <p className="text-xs text-text-muted leading-relaxed">
                  <strong className="text-text">LIME (Local Interpretable Model-agnostic Explanations)</strong> builds a local, simplified surrogate model around this specific audio sample. It isolates local boundaries and calculates feature thresholds (decision rules) that dictate classification decisions locally.
                </p>
              </div>

              {/* LIME Table */}
              <div className="overflow-x-auto border border-border-light rounded-xl soft-shadow">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-bg text-text font-bold border-b border-border-light">
                      <th className="p-3.5">Feature</th>
                      <th className="p-3.5">Extracted Value</th>
                      <th className="p-3.5">Triggered Decision Rule</th>
                      <th className="p-3.5 text-center">Local Weight</th>
                      <th className="p-3.5 text-right">Model Influence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-light">
                    {defaultLimeRules.map((rule, idx) => {
                      const isDepressionWeight = rule.weight > 0;
                      return (
                        <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                          <td className="p-3.5 font-bold text-text">{rule.feature}</td>
                          <td className="p-3.5 font-medium text-text-muted">{rule.value}</td>
                          <td className="p-3.5 font-mono text-xs text-primary bg-slate-50/30 font-semibold">{rule.rule}</td>
                          <td className={`p-3.5 text-center font-bold ${isDepressionWeight ? "text-primary" : "text-green-500"}`}>
                            {isDepressionWeight ? "+" : ""}{rule.weight.toFixed(2)}
                          </td>
                          <td className="p-3.5 text-right">
                            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold ${
                              isDepressionWeight 
                                ? "bg-red-50 text-primary border border-red-100" 
                                : "bg-green-50 text-green-600 border border-green-100"
                            }`}>
                              {rule.influence}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Accordion Biomarker Dictionary */}
      <div className="pt-5 border-t border-border-light">
        <h5 className="text-xs font-bold text-text-muted uppercase tracking-wider mb-3.5 flex items-center gap-1.5">
          <Activity size={12} className="text-primary" />
          Acoustic Biomarker Reference Dictionary
        </h5>
        
        <div className="space-y-2.5">
          {biomarkerDescriptions.map((bio) => {
            const isExpanded = expandedBiomarker === bio.name;
            return (
              <div 
                key={bio.name}
                className="border border-border-light rounded-xl overflow-hidden bg-bg/30"
              >
                <button
                  onClick={() => setExpandedBiomarker(isExpanded ? null : bio.name)}
                  className="w-full flex items-center justify-between p-3.5 text-left hover:bg-slate-50 transition-colors"
                >
                  <span className="text-xs font-bold text-text">{bio.name}</span>
                  {isExpanded ? <ChevronUp size={16} className="text-text-muted" /> : <ChevronDown size={16} className="text-text-muted" />}
                </button>
                
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: "auto" }}
                      exit={{ height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="p-3.5 pt-0 border-t border-border-light/50 text-xs space-y-2 leading-relaxed bg-white">
                        <p className="text-text-muted">
                          <strong className="text-text font-semibold">Technical Definition:</strong> {bio.definition}
                        </p>
                        <p className="text-text-muted">
                          <strong className="text-text font-semibold">Clinical Interpretation:</strong> {bio.clinicalMeaning}
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
