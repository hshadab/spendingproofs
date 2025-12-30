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
} from '@/components/landing';

type TabType = 'arc' | 'proof' | 'integrate';

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
          <UseCasesSection />
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

      {/* Footer - Always visible */}
      <FooterSection />
    </div>
  );
}
