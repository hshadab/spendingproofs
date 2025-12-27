'use client';

import { useState, ReactNode } from 'react';
import { Bot, Eye, Lock, Shield, Zap, ArrowRight } from 'lucide-react';

interface SplitDemoLayoutProps {
  agentView: ReactNode;
  verifierView: ReactNode;
  proofGenerated?: boolean;
}

export function SplitDemoLayout({ agentView, verifierView, proofGenerated }: SplitDemoLayoutProps) {
  const [showLabels, setShowLabels] = useState(true);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-bold">Two-Party Protocol View</h2>
          <button
            onClick={() => setShowLabels(!showLabels)}
            className="text-xs text-gray-400 hover:text-white transition-colors"
          >
            {showLabels ? 'Hide' : 'Show'} labels
          </button>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Lock className="w-3 h-3" />
          <span>Private data never leaves agent</span>
        </div>
      </div>

      {/* Split Panes */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Agent Pane */}
        <div className="relative">
          {showLabels && (
            <div className="absolute -top-3 left-4 z-10 bg-purple-500 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
              <Bot className="w-3 h-3" />
              Agent View
            </div>
          )}
          <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-5 h-full">
            {agentView}
          </div>
        </div>

        {/* Arrow */}
        <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20">
          {proofGenerated && (
            <div className="flex items-center gap-2 bg-[#0a0a0a] px-3 py-2 rounded-full border border-green-500/30">
              <Zap className="w-4 h-4 text-green-400" />
              <ArrowRight className="w-4 h-4 text-green-400" />
            </div>
          )}
        </div>

        {/* Verifier Pane */}
        <div className="relative">
          {showLabels && (
            <div className="absolute -top-3 left-4 z-10 bg-cyan-500 text-black text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
              <Eye className="w-3 h-3" />
              Verifier View
            </div>
          )}
          <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-5 h-full">
            {verifierView}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-lg p-4">
        <div className="grid md:grid-cols-2 gap-4 text-xs">
          <div>
            <h4 className="font-semibold text-purple-400 mb-2 flex items-center gap-1">
              <Bot className="w-3 h-3" />
              Agent Sees (Private)
            </h4>
            <ul className="space-y-1 text-gray-400">
              <li>- Policy thresholds and weights</li>
              <li>- Private context inputs</li>
              <li>- Full spending model parameters</li>
              <li>- Proof generation details</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-cyan-400 mb-2 flex items-center gap-1">
              <Eye className="w-3 h-3" />
              Verifier Sees (Public)
            </h4>
            <ul className="space-y-1 text-gray-400">
              <li>- policyId (identifier only)</li>
              <li>- modelHash, vkHash (commitments)</li>
              <li>- txIntentHash (transaction binding)</li>
              <li>- Decision output (approve/reject)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
