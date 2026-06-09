"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import Link from "next/link";

const navItems = [
  { name: "Home", href: "/#home" },
  { name: "Results", href: "/results" }, // will be dynamic
  { name: "AI Insights", href: "/#features" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [activeItem, setActiveItem] = useState("Home");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    
    // Check active result from localStorage dynamically
    const checkActiveId = () => {
      const id = localStorage.getItem("mindvoice_active_result_id");
      setActiveId(id);
    };
    checkActiveId();
    window.addEventListener("storage", checkActiveId);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("storage", checkActiveId);
    };
  }, []);

  const getResultsHref = () => {
    return activeId ? `/results?id=${activeId}` : "/results";
  };

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "glass-navbar shadow-sm border-b border-white/50"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-20">
          {/* Logo */}
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

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const href = item.name === "Results" ? getResultsHref() : item.href;
              return (
                <Link
                  key={item.name}
                  href={href}
                  onClick={() => setActiveItem(item.name)}
                  className={`relative px-5 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                    activeItem === item.name
                      ? "text-white gradient-bg shadow-md"
                      : "text-text-muted hover:text-text hover:bg-black/[0.03]"
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-xl hover:bg-black/5 transition-colors"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden pb-4 space-y-1"
          >
            {navItems.map((item) => {
              const href = item.name === "Results" ? getResultsHref() : item.href;
              return (
                <Link
                  key={item.name}
                  href={href}
                  onClick={() => {
                    setActiveItem(item.name);
                    setMobileOpen(false);
                  }}
                  className={`block px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    activeItem === item.name
                      ? "text-white gradient-bg"
                      : "text-text-muted hover:bg-black/[0.03]"
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </motion.div>
        )}
      </div>
    </motion.nav>
  );
}
