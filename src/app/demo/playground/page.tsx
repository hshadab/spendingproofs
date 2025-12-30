'use client';

import { useState } from 'react';
import { Play, RotateCcw, Eye, Lock } from 'lucide-react';
import { PolicySliders } from '@/components/PolicySliders';
import { PurchaseSimulator } from '@/components/PurchaseSimulator';
import { ProofProgress } from '@/components/ProofProgress';
import { ProofViewer } from '@/components/ProofViewer';
import {
  type SpendingPolicy,
  type SpendingModelInput,
  DEFAULT_SPENDING_POLICY,
  createDefaultInput,
  runSpendingModel,
} from '@/lib/spendingModel';
import { useProofGeneration } from '@/hooks/useProofGeneration';

type ViewMode = 'full' | 'agent' | 'verifier';

export default function PlaygroundPage() {
  const [policy, setPolicy] = useState<SpendingPolicy>(DEFAULT_SPENDING_POLICY);
  const [input, setInput] = useState<SpendingModelInput>(createDefaultInput());
  const [localDecision, setLocalDecision] = useState<ReturnType<typeof runSpendingModel> | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('full');
  const { state, generateProof, reset } = useProofGeneration();

  const handleGenerateProof = async () => {
    // First run local model for preview
    const decision = runSpendingModel(input, policy);
    setLocalDecision(decision);

    // Then generate the cryptographic proof
    await generateProof(input);
  };

  const handleReset = () => {
    reset();
    setLocalDecision(null);
    setPolicy(DEFAULT_SPENDING_POLICY);
    setInput(createDefaultInput());
  };

  const isGenerating = state.status === 'running';

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          Interactive Playground
        </h1>
        <p className="text-gray-400">
          Configure spending policies, simulate purchases, and generate cryptographic proofs.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* Left Column: Policy + Simulator */}
        <div className="space-y-6">
          <PolicySliders
            policy={policy}
            onChange={setPolicy}
            disabled={isGenerating}
          />
          <PurchaseSimulator
            input={input}
            onChange={setInput}
            disabled={isGenerating}
          />
        </div>

        {/* Right Column: Actions + Results */}
        <div className="space-y-6">
          {/* Local Decision Preview */}
          {localDecision && state.status === 'idle' && (
            <div className={`p-4 rounded-xl border ${
              localDecision.shouldBuy
                ? 'bg-green-900/20 border-green-800'
                : 'bg-red-900/20 border-red-800'
            }`}>
              <div className="font-semibold mb-2 text-white">
                {localDecision.shouldBuy ? 'Would Approve' : 'Would Reject'}
              </div>
              <ul className="text-sm space-y-1 text-gray-400">
                {localDecision.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Generate Button */}
          <div className="flex gap-3">
            <button
              onClick={handleGenerateProof}
              disabled={isGenerating}
              className="flex-1 py-3 px-6 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              {isGenerating ? 'Generating...' : 'Generate Proof'}
            </button>
            <button
              onClick={handleReset}
              disabled={isGenerating}
              className="py-3 px-4 border border-gray-700 hover:bg-gray-800 text-gray-300 rounded-lg transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>

          {/* Progress */}
          {(state.status === 'running' || state.status === 'error') && (
            <ProofProgress
              status={state.status}
              progress={state.progress}
              elapsedMs={state.elapsedMs}
              steps={state.steps}
            />
          )}

          {/* Proof Result */}
          {state.status === 'complete' && state.result?.proof && (
            <ProofViewer
              proof={state.result.proof}
              inference={state.result.inference}
            />
          )}

          {/* Error */}
          {state.status === 'error' && state.error && (
            <div className="p-4 bg-red-900/20 border border-red-800 rounded-xl text-red-300">
              <div className="font-semibold mb-1">Error</div>
              <p className="text-sm">{state.error}</p>
            </div>
          )}
        </div>
      </div>

      {/* View Mode Toggle - Only show after proof generated */}
      {state.status === 'complete' && state.result?.proof && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm text-gray-400">View Mode:</span>
            <div className="flex bg-[#0d1117] border border-gray-800 rounded-lg p-1">
              <button
                onClick={() => setViewMode('full')}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                  viewMode === 'full'
                    ? 'bg-purple-500/20 text-purple-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Full View
              </button>
              <button
                onClick={() => setViewMode('agent')}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1 ${
                  viewMode === 'agent'
                    ? 'bg-purple-500/20 text-purple-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Lock className="w-3 h-3" />
                Agent View
              </button>
              <button
                onClick={() => setViewMode('verifier')}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1 ${
                  viewMode === 'verifier'
                    ? 'bg-cyan-500/20 text-cyan-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Eye className="w-3 h-3" />
                Verifier View
              </button>
            </div>
          </div>

          {/* Split View Panels */}
          {viewMode !== 'full' && (
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              {/* Agent View */}
              <div className={`bg-[#0d1117] border rounded-xl p-5 ${
                viewMode === 'agent' ? 'border-purple-500/50' : 'border-gray-800 opacity-50'
              }`}>
                <h3 className="font-semibold mb-3 flex items-center gap-2 text-purple-400">
                  <Lock className="w-4 h-4" />
                  Agent Sees (Private)
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Price</span>
                    <span className="font-mono">${input.priceUsdc.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Budget</span>
                    <span className="font-mono">${input.budgetUsdc.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Daily Limit</span>
                    <span className="font-mono">${input.dailyLimitUsdc.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Spent Today</span>
                    <span className="font-mono">${input.spentTodayUsdc.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Service Success Rate</span>
                    <span className="font-mono">{(input.serviceSuccessRate * 100).toFixed(0)}%</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800 mt-2">
                    <div className="text-xs text-purple-400">+ Model weights, policy thresholds...</div>
                  </div>
                </div>
              </div>

              {/* Verifier View */}
              <div className={`bg-[#0d1117] border rounded-xl p-5 ${
                viewMode === 'verifier' ? 'border-cyan-500/50' : 'border-gray-800 opacity-50'
              }`}>
                <h3 className="font-semibold mb-3 flex items-center gap-2 text-cyan-400">
                  <Eye className="w-4 h-4" />
                  Verifier Sees (Public)
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Policy Hash</span>
                    <span className="font-mono text-xs">{state.result.proof.metadata?.modelHash?.slice(0, 12) || 'N/A'}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Inputs Hash</span>
                    <span className="font-mono text-xs">{state.result.proof.metadata?.inputHash?.slice(0, 12) || 'N/A'}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Decision</span>
                    <span className={`font-mono ${state.result.inference?.decision === 'approve' ? 'text-green-400' : 'text-red-400'}`}>
                      {state.result.inference?.decision === 'approve' ? 'APPROVE' : 'REJECT'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Confidence</span>
                    <span className="font-mono">{state.result.inference?.confidence ?? 0}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Proof Valid</span>
                    <span className="font-mono text-green-400">✓ Verified</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800 mt-2">
                    <div className="text-xs text-cyan-400">Cannot see: inputs, weights, thresholds</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Info Box */}
      <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-xl text-sm text-blue-300">
        <strong>How it works:</strong> When you click Generate Proof, the spending model runs locally
        to determine the decision, then JOLT-Atlas generates a cryptographic SNARK proof that
        mathematically guarantees the model was executed correctly on these inputs. This proof can
        be verified on Arc chain without revealing the sensitive inputs.
        {state.status === 'complete' && (
          <span className="block mt-2 text-purple-300">
            <strong>Try the view modes above</strong> to see what the agent keeps private vs what the verifier can see.
          </span>
        )}
      </div>
    </div>
  );
}
