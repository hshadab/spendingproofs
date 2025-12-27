'use client';

import { useState } from 'react';
import { Play, RotateCcw } from 'lucide-react';
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

export default function PlaygroundPage() {
  const [policy, setPolicy] = useState<SpendingPolicy>(DEFAULT_SPENDING_POLICY);
  const [input, setInput] = useState<SpendingModelInput>(createDefaultInput());
  const [localDecision, setLocalDecision] = useState<ReturnType<typeof runSpendingModel> | null>(null);
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

      {/* Info Box */}
      <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-xl text-sm text-blue-300">
        <strong>How it works:</strong> When you click Generate Proof, the spending model runs locally
        to determine the decision, then JOLT-Atlas generates a cryptographic SNARK proof that
        mathematically guarantees the model was executed correctly on these inputs. This proof can
        be verified on Arc chain without revealing the sensitive inputs.
      </div>
    </div>
  );
}
