"use client";

import { motion } from "framer-motion";

export default function Footer() {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="w-full border-t border-border-light bg-white/50"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center">
              <div className="w-3.5 h-3.5 bg-white/30 rounded-full" />
            </div>
            <div>
              <span className="font-bold text-text text-sm">MindVoice AI</span>
              <span className="text-text-muted text-xs ml-2">
                © {new Date().getFullYear()} Telkom University
              </span>
            </div>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6">
            {["Privacy", "Terms", "Contact"].map((item) => (
              <a
                key={item}
                href="#"
                className="text-sm text-text-muted hover:text-primary transition-colors"
              >
                {item}
              </a>
            ))}
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
