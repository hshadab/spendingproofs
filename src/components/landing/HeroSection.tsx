'use client';

import Link from 'next/link';
import { Shield, ArrowRight } from 'lucide-react';
import { InstallCommand } from './utils';
import { HeroFlowDiagram } from './HeroFlowDiagram';

interface HeroSectionProps {
  onGetStarted: () => void;
}

export function HeroSection({ onGetStarted }: HeroSectionProps) {
  return (
    <section className="pt-24 pb-16 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Left: Title and CTA */}
          <div>
            {/* Badges */}
            <div className="flex items-center gap-3 mb-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-500/10 border border-purple-500/30 rounded-full text-sm text-purple-400">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                Built for Arc Agentic Commerce
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-sm text-cyan-400">
                <Shield className="w-3 h-3" />
                Testnet Alpha
              </div>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">
                Spending Guardrails
              </span>
              <br />
              for Agentic Commerce
            </h1>
            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              <strong className="text-white">A cryptographic proof primitive for Arc.</strong>{' '}
              Enable AI agents to spend USDC with proof of policy compliance.
              No trust required—just math. The infrastructure for machine-to-machine commerce.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 mb-8">
              <button
                onClick={onGetStarted}
                className="inline-flex items-center justify-center gap-2 bg-white text-black px-6 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors"
              >
                Get Started
                <ArrowRight className="w-4 h-4" />
              </button>
              <Link
                href="/demo"
                className="inline-flex items-center justify-center gap-2 border border-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-white/5 transition-colors"
              >
                View Demo
              </Link>
            </div>

            {/* Install command */}
            <div className="max-w-md">
              <InstallCommand pkg="@hshadab/spending-proofs" />
            </div>
          </div>

          {/* Right: Flow Diagram */}
          <div className="flex justify-center lg:justify-center">
            <HeroFlowDiagram />
          </div>
        </div>
      </div>
    </section>
  );
}
