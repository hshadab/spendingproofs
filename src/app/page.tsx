'use client';

import { useState, useMemo } from 'react';
import {
  Navigation,
  HeroSection,
  ProblemSection,
  SolutionSection,
  WhyArcSection,
  UseCasesSection,
  ProofSystemSection,
  AgentFlowSection,
  IntegrateSection,
  FooterSection,
} from '@/components/landing';
import { useScrollSpy } from '@/hooks/useScrollSpy';

type TabType = 'arc' | 'proof' | 'integrate' | 'deep-dive';

// Section IDs for each tab
const tabSections: Record<TabType, string[]> = {
  arc: ['problem', 'solution', 'why-arc'],
  proof: ['whats-proven', 'model', 'jolt-atlas'],
  integrate: ['architecture', 'install-sdk', 'deploy', 'roadmap'],
  'deep-dive': ['agent-flow', 'use-cases', 'chain-comparison'],
};

export default function Home() {
  const [activeSection, setActiveSection] = useState<TabType>('arc');

  // Get the section IDs for the current tab
  const currentSections = useMemo(() => tabSections[activeSection], [activeSection]);

  // Track which section is currently in view
  const activeScrollSection = useScrollSpy(currentSections, 100);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <Navigation
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        activeScrollSection={activeScrollSection}
      />

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
                Explore the agent transaction flow, use cases, and chain comparisons.
              </p>
            </div>
          </section>
          <AgentFlowSection />
          <UseCasesSection />
        </>
      )}

      {/* Footer - Always visible */}
      <FooterSection />
    </div>
  );
}
