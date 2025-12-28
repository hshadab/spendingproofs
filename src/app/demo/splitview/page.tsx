'use client';

import { useState } from 'react';
import { Users } from 'lucide-react';
import { SplitDemoLayout } from '@/components/SplitDemoLayout';
import { AgentPane } from '@/components/AgentPane';
import { VerifierPane } from '@/components/VerifierPane';
import { PerformanceMetrics } from '@/components/PerformanceMetrics';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import type { SpendingModelInput } from '@/lib/spendingModel';

export default function SplitViewPage() {
  const { state, generateProof, reset } = useProofGeneration();
  const [proofResult, setProofResult] = useState<typeof state.result>(undefined);

  const handleGenerate = async (inputs: SpendingModelInput) => {
    const result = await generateProof(inputs);
    if (result.success) {
      setProofResult(result);
    }
  };

  const handleReset = () => {
    reset();
    setProofResult(undefined);
  };

  return (
    <div className="max-w-6xl mx-auto py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/20 rounded-full px-3 py-1 text-sm text-purple-400 mb-4">
          <Users className="w-4 h-4" />
          Two-Party Protocol
        </div>
        <h1 className="text-3xl font-bold mb-3">Agent vs Verifier Split View</h1>
        <p className="text-gray-400 max-w-2xl">
          This demo shows the two-party nature of spending proofs. The <strong className="text-purple-400">agent</strong> sees
          all private data, while the <strong className="text-cyan-400">verifier</strong> only sees public signals and the proof.
        </p>
      </div>

      {/* Reset Button */}
      {proofResult && (
        <div className="flex justify-end mb-4">
          <button
            onClick={handleReset}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            Reset Demo
          </button>
        </div>
      )}

      {/* Split View */}
      <SplitDemoLayout
        agentView={
          <AgentPane
            isGenerating={state.status === 'running'}
            onGenerate={handleGenerate}
          />
        }
        verifierView={
          <VerifierPane
            proof={proofResult}
            isWaiting={state.status === 'running'}
          />
        }
        proofGenerated={!!proofResult}
      />

      {/* Progress */}
      {state.status === 'running' && (
        <div className="mt-8 bg-[#0d1117] border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm text-gray-400">Generating proof...</span>
            <span className="text-sm font-mono text-white">{(state.elapsedMs / 1000).toFixed(1)}s</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2 mb-4">
            <div
              className="bg-purple-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${state.progress}%` }}
            />
          </div>
          <div className="space-y-2">
            {state.steps.map((step, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                {step.status === 'done' && <span className="text-green-400">✓</span>}
                {step.status === 'running' && (
                  <div className="w-3 h-3 border border-purple-500 border-t-transparent rounded-full animate-spin" />
                )}
                {step.status === 'pending' && <span className="text-gray-600">○</span>}
                <span className={`${
                  step.status === 'done' ? 'text-gray-400' :
                  step.status === 'running' ? 'text-white' :
                  'text-gray-600'
                }`}>
                  {step.name}
                </span>
                {step.durationMs && (
                  <span className="text-gray-500 ml-auto">{step.durationMs}ms</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Insight */}
      <div className="mt-8 bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-500/20 rounded-xl p-6">
        <h3 className="font-semibold mb-3">Why This Matters</h3>
        <p className="text-sm text-gray-400 mb-4">
          Traditional financial systems require revealing sensitive data to verify compliance.
          With spending proofs:
        </p>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-[#0a0a0a]/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-medium mb-2">Agent Keeps Private</h4>
            <ul className="space-y-1 text-gray-400">
              <li>- Exact budget thresholds</li>
              <li>- Spending history and patterns</li>
              <li>- Decision model weights</li>
              <li>- Policy configuration details</li>
            </ul>
          </div>
          <div className="bg-[#0a0a0a]/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-medium mb-2">Verifier Can Trust</h4>
            <ul className="space-y-1 text-gray-400">
              <li>- Policy was properly evaluated</li>
              <li>- Model matches registered policy</li>
              <li>- Proof is bound to this tx intent</li>
              <li>- Decision is mathematically sound</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Performance */}
      <div className="mt-8">
        <PerformanceMetrics variant="compact" />
      </div>
    </div>
  );
}
