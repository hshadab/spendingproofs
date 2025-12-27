'use client';

import { Check, X, Copy, ExternalLink, Shield } from 'lucide-react';
import { truncateHash } from '@/lib/contracts';
import type { ProofData, InferenceResult } from '@/lib/types';
import { useState } from 'react';

interface ProofViewerProps {
  proof: ProofData;
  inference?: InferenceResult;
}

export function ProofViewer({ proof, inference }: ProofViewerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(proof.proof);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isApproved = inference?.decision === 'approve';

  return (
    <div className="bg-[#0d1117] rounded-xl border border-gray-800 border-gray-800 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-purple-600" />
          Proof Generated
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-400 text-gray-400 hover:bg-slate-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <a
            href={`https://testnet.arcscan.app`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-purple-600 hover:bg-purple-50 dark:hover:bg-purple-900/20 rounded-lg transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Arc Explorer
          </a>
        </div>
      </div>

      {/* Decision */}
      {inference && (
        <div
          className={`mb-6 p-4 rounded-lg flex items-center gap-4 ${
            isApproved
              ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 border-green-800'
              : 'bg-red-50 dark:bg-red-900/20 border border-red-200 border-red-800'
          }`}
        >
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center ${
              isApproved ? 'bg-green-100 dark:bg-green-900/40' : 'bg-red-100 dark:bg-red-900/40'
            }`}
          >
            {isApproved ? (
              <Check className="w-6 h-6 text-green-600 text-green-400" />
            ) : (
              <X className="w-6 h-6 text-red-600 text-red-400" />
            )}
          </div>
          <div>
            <div
              className={`text-lg font-semibold ${
                isApproved
                  ? 'text-green-700 text-green-300'
                  : 'text-red-700 text-red-300'
              }`}
            >
              {isApproved ? 'APPROVE PURCHASE' : 'REJECT PURCHASE'}
            </div>
            <div className="text-sm text-gray-400 text-gray-400">
              Confidence: {(inference.confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      )}

      {/* Proof Metadata */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
          Proof Metadata
        </h4>

        <div className="grid gap-3 text-sm">
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Proof Hash</span>
            <span className="font-mono text-white text-white">
              {truncateHash(proof.proofHash)}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Model Hash</span>
            <span className="font-mono text-white text-white">
              {truncateHash(proof.metadata.modelHash)}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Input Hash</span>
            <span className="font-mono text-white text-white">
              {truncateHash(proof.metadata.inputHash)}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Output Hash</span>
            <span className="font-mono text-white text-white">
              {truncateHash(proof.metadata.outputHash)}
            </span>
          </div>
          {proof.metadata.txIntentHash && (
            <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
              <span className="text-gray-500 flex items-center gap-1">
                txIntentHash
                <span className="text-xs text-purple-400">(binding)</span>
              </span>
              <span className="font-mono text-purple-400">
                {truncateHash(proof.metadata.txIntentHash)}
              </span>
            </div>
          )}
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Proof Size</span>
            <span className="font-mono text-white text-white">
              {(proof.metadata.proofSize / 1024).toFixed(1)} KB
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-slate-100 border-gray-800">
            <span className="text-gray-500">Generation Time</span>
            <span className="font-mono text-white text-white">
              {(proof.metadata.generationTime / 1000).toFixed(2)}s
            </span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-500">Prover</span>
            <span className="font-mono text-white text-white text-xs">
              {proof.metadata.proverVersion}
            </span>
          </div>
        </div>
      </div>

      {/* Raw Output */}
      {inference?.rawOutput && (
        <div className="mt-6 p-4 bg-gray-800 bg-800 rounded-lg">
          <div className="text-xs text-gray-500 mb-2">Raw Model Output</div>
          <code className="text-sm font-mono text-slate-700 text-slate-300">
            [{inference.rawOutput.map((v) => v.toFixed(4)).join(', ')}]
          </code>
          <div className="text-xs text-gray-400 mt-1">
            (shouldBuy, confidence, riskScore)
          </div>
        </div>
      )}
    </div>
  );
}
