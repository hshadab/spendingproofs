'use client';

import { useState } from 'react';
import { Lock, Bot, DollarSign, Shield, Clock, TrendingUp, Play } from 'lucide-react';
import type { SpendingModelInput } from '@/lib/spendingModel';
import type { ProveResponse } from '@/lib/types';

interface AgentPaneProps {
  onProofGenerated?: (result: ProveResponse, inputs: SpendingModelInput) => void;
  isGenerating?: boolean;
  onGenerate?: (inputs: SpendingModelInput) => void;
}

export function AgentPane({ onProofGenerated, isGenerating, onGenerate }: AgentPaneProps) {
  const [inputs, setInputs] = useState<SpendingModelInput>({
    serviceUrl: 'https://api.example.com/v1',
    serviceName: 'Example API Service',
    serviceCategory: 'api',
    priceUsdc: 0.05,
    budgetUsdc: 1.0,
    spentTodayUsdc: 0.2,
    dailyLimitUsdc: 0.5,
    serviceSuccessRate: 0.95,
    serviceTotalCalls: 100,
    purchasesInCategory: 5,
    timeSinceLastPurchase: 2.5,
  });

  const handleGenerate = () => {
    onGenerate?.(inputs);
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold">Agent Private Context</h3>
        </div>
        <div className="flex items-center gap-1 text-xs text-purple-400">
          <Lock className="w-3 h-3" />
          <span>Private</span>
        </div>
      </div>

      {/* Policy Parameters (Private) */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wide flex items-center gap-1">
          <Shield className="w-3 h-3" />
          Policy Parameters (Hidden from Verifier)
        </h4>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Daily Limit</label>
            <div className="flex items-center gap-2">
              <DollarSign className="w-3 h-3 text-green-400" />
              <input
                type="number"
                step="0.1"
                value={inputs.dailyLimitUsdc}
                onChange={(e) => setInputs({ ...inputs, dailyLimitUsdc: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-16 outline-none"
              />
              <span className="text-xs text-gray-500">USDC</span>
            </div>
          </div>

          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Budget Remaining</label>
            <div className="flex items-center gap-2">
              <DollarSign className="w-3 h-3 text-green-400" />
              <input
                type="number"
                step="0.1"
                value={inputs.budgetUsdc}
                onChange={(e) => setInputs({ ...inputs, budgetUsdc: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-16 outline-none"
              />
              <span className="text-xs text-gray-500">USDC</span>
            </div>
          </div>
        </div>
      </div>

      {/* Transaction Context (Private) */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wide flex items-center gap-1">
          <DollarSign className="w-3 h-3" />
          Transaction Context (Hidden from Verifier)
        </h4>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Purchase Price</label>
            <div className="flex items-center gap-2">
              <DollarSign className="w-3 h-3 text-amber-400" />
              <input
                type="number"
                step="0.01"
                value={inputs.priceUsdc}
                onChange={(e) => setInputs({ ...inputs, priceUsdc: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-16 outline-none"
              />
              <span className="text-xs text-gray-500">USDC</span>
            </div>
          </div>

          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Spent Today</label>
            <div className="flex items-center gap-2">
              <DollarSign className="w-3 h-3 text-amber-400" />
              <input
                type="number"
                step="0.05"
                value={inputs.spentTodayUsdc}
                onChange={(e) => setInputs({ ...inputs, spentTodayUsdc: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-16 outline-none"
              />
              <span className="text-xs text-gray-500">USDC</span>
            </div>
          </div>
        </div>
      </div>

      {/* Behavioral Context (Private) */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wide flex items-center gap-1">
          <TrendingUp className="w-3 h-3" />
          Behavioral Context (Hidden from Verifier)
        </h4>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Service Success Rate</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={inputs.serviceSuccessRate}
                onChange={(e) => setInputs({ ...inputs, serviceSuccessRate: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-12 outline-none"
              />
              <span className="text-xs text-gray-500">(0-1)</span>
            </div>
          </div>

          <div className="bg-[#0a0a0a] p-3 rounded-lg border border-purple-500/20">
            <label className="text-xs text-gray-500 block mb-1">Time Since Last</label>
            <div className="flex items-center gap-2">
              <Clock className="w-3 h-3 text-blue-400" />
              <input
                type="number"
                step="0.5"
                value={inputs.timeSinceLastPurchase}
                onChange={(e) => setInputs({ ...inputs, timeSinceLastPurchase: parseFloat(e.target.value) || 0 })}
                className="bg-transparent text-white font-mono text-sm w-12 outline-none"
              />
              <span className="text-xs text-gray-500">hrs</span>
            </div>
          </div>
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-600/50 px-4 py-3 rounded-lg font-medium transition-colors"
      >
        {isGenerating ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Generating Proof...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            Generate Proof
          </>
        )}
      </button>

      {/* Privacy Note */}
      <p className="text-xs text-gray-500 text-center">
        All values above stay private. Only the proof and public signals are shared.
      </p>
    </div>
  );
}
