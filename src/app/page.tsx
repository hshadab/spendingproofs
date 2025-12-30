'use client';

import { useState } from 'react';
import {
  Navigation,
  HeroSection,
  ProblemSection,
  SolutionSection,
  WhyArcSection,
  UseCasesSection,
  ProofSystemSection,
  IntegrateSection,
  FooterSection,
  ComparisonTable,
} from '@/components/landing';

type TabType = 'arc' | 'proof' | 'integrate' | 'deep-dive';

export default function Home() {
  const [activeSection, setActiveSection] = useState<TabType>('arc');

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <Navigation activeSection={activeSection} setActiveSection={setActiveSection} />

      {/* === ARC + AGENTS TAB === */}
      {activeSection === 'arc' && (
        <>
          <HeroSection onGetStarted={() => setActiveSection('integrate')} />
          <ProblemSection />
          <SolutionSection />
          <WhyArcSection />
        </>
      )}

      {/* === PROOF SYSTEM TAB === */}
      {activeSection === 'proof' && (
        <ProofSystemSection />
      )}

      {/* === INTEGRATE TAB === */}
      {activeSection === 'integrate' && (
        <IntegrateSection />
      )}

      {/* === DEEPER DIVE TAB === */}
      {activeSection === 'deep-dive' && (
        <>
          <section className="pt-24 pb-8 px-6">
            <div className="max-w-6xl mx-auto text-center">
              <h1 className="text-4xl font-bold mb-4">Deeper Dive</h1>
              <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                Explore use cases, chain comparisons, and detailed analysis of why spending proofs matter for the agent economy.
              </p>
            </div>
          </section>
          <UseCasesSection />
          <section id="chain-comparison" className="py-16 px-6 border-t border-gray-800">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold mb-4">Chain Comparison</h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                  See how Arc&apos;s infrastructure compares to other L2s for autonomous agent operations.
                </p>
              </div>
              <ComparisonTable />
            </div>
          </section>
        </>
      )}

      {/* Footer - Always visible */}
      <FooterSection />
    </div>
  );
}
