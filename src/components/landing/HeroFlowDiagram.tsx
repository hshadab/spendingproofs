'use client';

import { Shield, ArrowRight, Check, X } from 'lucide-react';

export function HeroFlowDiagram() {
  return (
    <div className="relative">
      {/* Glow effect */}
      <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-purple-500 to-cyan-500 rounded-xl blur-lg opacity-30" />

      <div className="relative bg-[#0d1117] border border-purple-500/30 rounded-xl p-6 w-full max-w-md shadow-[0_0_30px_rgba(168,85,247,0.15)]">
        {/* Header */}
        <div className="text-center mb-6">
          <h4 className="text-sm font-semibold text-gray-300 mb-1">How It Works</h4>
          <p className="text-xs text-gray-500">Proof-gated USDC payments on Arc</p>
        </div>

        {/* Flow Steps */}
        <div className="space-y-4">
          {/* Step 1: Agent */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-lg">🤖</span>
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-white">Agent requests $0.05</div>
              <div className="text-xs text-gray-500">Wants to buy an API call</div>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex items-center justify-center">
            <ArrowRight className="w-4 h-4 text-gray-600" />
          </div>

          {/* Step 2: Proof Generation */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Shield className="w-5 h-5 text-cyan-400" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-white">Generates zkML proof</div>
              <div className="text-xs text-gray-500">"I evaluated my spending policy"</div>
            </div>
            <div className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded">
              48KB
            </div>
          </div>

          {/* Arrow */}
          <div className="flex items-center justify-center">
            <ArrowRight className="w-4 h-4 text-gray-600" />
          </div>

          {/* Step 3: Verification Gate */}
          <div className="bg-gradient-to-r from-amber-900/20 to-amber-900/10 border border-amber-500/30 rounded-lg p-3">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-sm">🔐</span>
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium text-amber-400">Verification Gate</div>
                <div className="text-xs text-gray-500">Merchant/protocol checks proof</div>
              </div>
            </div>

            {/* Two outcomes */}
            <div className="grid grid-cols-2 gap-2 mt-3">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-2 text-center">
                <Check className="w-4 h-4 text-green-400 mx-auto mb-1" />
                <div className="text-xs text-green-400 font-medium">Valid</div>
                <div className="text-[10px] text-gray-500">USDC sent</div>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-2 text-center">
                <X className="w-4 h-4 text-red-400 mx-auto mb-1" />
                <div className="text-xs text-red-400 font-medium">Invalid</div>
                <div className="text-[10px] text-gray-500">TX rejected</div>
              </div>
            </div>
          </div>
        </div>

        {/* What's proven */}
        <div className="mt-5 pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-500 mb-2">The proof guarantees:</div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-3 h-3 text-green-400 flex-shrink-0" />
              <span className="text-gray-400">Policy model was actually run</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-3 h-3 text-green-400 flex-shrink-0" />
              <span className="text-gray-400">Inputs weren't tampered with</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-3 h-3 text-green-400 flex-shrink-0" />
              <span className="text-gray-400">Decision matches model output</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
