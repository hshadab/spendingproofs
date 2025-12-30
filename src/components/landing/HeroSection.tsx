'use client';

import Link from 'next/link';
import { Shield, ArrowRight, ExternalLink } from 'lucide-react';
import { InstallCommand } from './utils';
import { ModelExplorer } from './ModelExplorer';

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
            {/* Built for Arc Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-500/10 border border-purple-500/30 rounded-full text-sm text-purple-400 mb-6">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
              Built for Arc
              <a
                href="https://arc.builders"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-purple-300 transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
              Prove Your Agent
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">
                Followed Its Policy.
              </span>
            </h1>
            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              <strong className="text-white">zkML spending proofs</strong> for autonomous agents on Arc.
              Cryptographically prove your agent evaluated its spending policy—without revealing the policy itself.
              Production infrastructure for agent commerce. Deploy today.
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

            {/* Value Props for Arc Builders Fund */}
            <div className="grid grid-cols-3 gap-4 mb-6 text-center">
              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="text-lg font-bold text-green-400">USDC</div>
                <div className="text-xs text-gray-500">Native Gas</div>
              </div>
              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="text-lg font-bold text-cyan-400">&lt;1s</div>
                <div className="text-xs text-gray-500">Finality</div>
              </div>
              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="text-lg font-bold text-purple-400">zkML</div>
                <div className="text-xs text-gray-500">Enforcement</div>
              </div>
            </div>

            {/* Install command */}
            <div className="max-w-md">
              <InstallCommand pkg="@icme-labs/spending-proofs" />
            </div>
          </div>

          {/* Right: Interactive Model Explorer */}
          <div className="flex justify-center lg:justify-center">
            <ModelExplorer />
          </div>
        </div>
      </div>
    </section>
  );
}
