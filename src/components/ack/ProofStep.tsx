'use client';

import { Lock, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import type { SpendingProof } from '@/lib/types';
import type { ProofStatus } from '@/lib/types';

interface ProofStepProps {
  proof: SpendingProof | null;
  status: ProofStatus;
  progress: number;
  elapsedMs: number;
  error: string | null;
  onGenerateProof: () => Promise<void>;
  disabled?: boolean;
}

export function ProofStep({
  proof,
  status,
  progress,
  elapsedMs,
  error,
  onGenerateProof,
  disabled,
}: ProofStepProps) {
  const isGenerating = status === 'running';

  if (proof) {
    return (
      <div className="bg-[#0d1117] border border-green-800/50 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">3. zkML Proof Generated</h3>
            <p className="text-sm text-green-400">Cryptographic proof ready</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Decision</span>
            <span className={`font-medium ${proof.decision.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
              {proof.decision.shouldBuy ? 'APPROVED' : 'REJECTED'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Confidence</span>
            <span className="text-white">{(proof.decision.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Risk Score</span>
            <span className="text-white">{(proof.decision.riskScore * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Proof Hash</span>
            <span className="text-cyan-400 font-mono text-sm">
              {proof.proofHash.slice(0, 10)}...{proof.proofHash.slice(-6)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Proof Size</span>
            <span className="text-white">{(proof.proofSizeBytes / 1024).toFixed(1)} KB</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Generation Time</span>
            <span className="text-white">{(proof.generationTimeMs / 1000).toFixed(1)}s</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0d1117] border border-purple-800/50 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
          <Lock className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">3. Generate zkML Proof</h3>
          <p className="text-sm text-gray-400">Create cryptographic spending proof</p>
        </div>
      </div>

      {isGenerating && (
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-400">Generating proof...</span>
            <span className="text-purple-400">{(elapsedMs / 1000).toFixed(1)}s</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div
              className="bg-purple-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}

      <button
        onClick={onGenerateProof}
        disabled={disabled || isGenerating}
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {isGenerating ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Generating Proof...
          </>
        ) : (
          <>
            <Lock className="w-5 h-5" />
            Generate zkML Proof
          </>
        )}
      </button>

      <p className="mt-4 text-xs text-gray-500">
        Generates a JOLT-Atlas SNARK proof that the spending policy was correctly evaluated.
      </p>
    </div>
  );
}
