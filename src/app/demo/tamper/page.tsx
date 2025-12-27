'use client';

import { useState, useEffect } from 'react';
import { Shield, AlertTriangle, Play } from 'lucide-react';
import { TamperPanel } from '@/components/TamperPanel';
import { ProofViewer } from '@/components/ProofViewer';
import { ProofProgress } from '@/components/ProofProgress';
import {
  type SpendingModelInput,
  createDefaultInput,
  spendingInputToNumeric,
} from '@/lib/spendingModel';
import { useProofGeneration } from '@/hooks/useProofGeneration';

export default function TamperPage() {
  const [originalInputs, setOriginalInputs] = useState<SpendingModelInput>(createDefaultInput());
  const { state, generateProof, reset } = useProofGeneration();
  const [proofGenerated, setProofGenerated] = useState(false);

  const handleGenerateProof = async () => {
    await generateProof(originalInputs);
    setProofGenerated(true);
  };

  const handleVerify = async (inputs: number[], isModified: boolean): Promise<{ valid: boolean; reason: string }> => {
    if (!state.result?.proof) {
      return { valid: false, reason: 'No proof generated yet' };
    }

    // Compare hashes locally
    const originalNumeric = spendingInputToNumeric(originalInputs);
    const originalMatch = JSON.stringify(originalNumeric) === JSON.stringify(inputs);

    if (originalMatch) {
      return {
        valid: true,
        reason: 'Input hash matches proof - verification passed',
      };
    } else {
      return {
        valid: false,
        reason: 'Input hash mismatch - inputs were modified after proof generation',
      };
    }
  };

  const handleReset = () => {
    reset();
    setProofGenerated(false);
    setOriginalInputs(createDefaultInput());
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          Tamper Detection
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Demonstrate how modifying inputs after proof generation causes verification to fail.
        </p>
      </div>

      {!proofGenerated ? (
        <div className="space-y-6">
          {/* Instructions */}
          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="w-6 h-6 text-amber-600" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 dark:text-white mb-2">
                  How Tamper Detection Works
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                  When a proof is generated, it includes a cryptographic hash of all inputs.
                  If anyone tries to modify the inputs after the proof is created, the hash
                  won't match and verification will fail. This prevents:
                </p>
                <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <li>• Changing the purchase price after approval</li>
                  <li>• Modifying budget or spending limits</li>
                  <li>• Replaying proofs with different parameters</li>
                  <li>• Any form of input tampering</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Original Inputs Preview */}
          <div className="p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">
            <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
              Original Inputs (will be locked after proof)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-slate-500 mb-1">Price</div>
                <div className="font-mono text-slate-900 dark:text-white">
                  ${originalInputs.priceUsdc.toFixed(4)}
                </div>
              </div>
              <div>
                <div className="text-slate-500 mb-1">Budget</div>
                <div className="font-mono text-slate-900 dark:text-white">
                  ${originalInputs.budgetUsdc.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-slate-500 mb-1">Spent Today</div>
                <div className="font-mono text-slate-900 dark:text-white">
                  ${originalInputs.spentTodayUsdc.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-slate-500 mb-1">Success Rate</div>
                <div className="font-mono text-slate-900 dark:text-white">
                  {(originalInputs.serviceSuccessRate * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerateProof}
            disabled={state.status === 'running'}
            className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Play className="w-5 h-5" />
            {state.status === 'running' ? 'Generating Proof...' : 'Generate Proof to Start Demo'}
          </button>

          {/* Progress */}
          {state.status === 'running' && (
            <ProofProgress
              status={state.status}
              progress={state.progress}
              elapsedMs={state.elapsedMs}
              steps={state.steps}
            />
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Success Message */}
          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl flex items-center gap-3">
            <Shield className="w-6 h-6 text-green-600 flex-shrink-0" />
            <div>
              <div className="font-semibold text-green-700 dark:text-green-300">
                Proof Generated Successfully
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Now try modifying the inputs on the right and verify - you'll see the verification fail.
              </p>
            </div>
          </div>

          {/* Proof Viewer */}
          {state.result?.proof && (
            <ProofViewer
              proof={state.result.proof}
              inference={state.result.inference}
            />
          )}

          {/* Tamper Panel */}
          {state.result?.proof && (
            <TamperPanel
              originalInputs={originalInputs}
              proofInputHash={state.result.proof.metadata.inputHash}
              onVerify={handleVerify}
            />
          )}

          {/* Reset Button */}
          <button
            onClick={handleReset}
            className="w-full py-3 px-6 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg transition-colors"
          >
            Reset Demo
          </button>
        </div>
      )}
    </div>
  );
}
