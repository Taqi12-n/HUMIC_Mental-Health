import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import UploadSection from "@/components/UploadSection";
import HowItWorks from "@/components/HowItWorks";
import Features from "@/components/Features";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Navbar />
      <main className="flex-1">
        <Hero />
        <UploadSection />
        <HowItWorks />
        <Features />
      </main>
      <Footer />
    </>
  );
}
