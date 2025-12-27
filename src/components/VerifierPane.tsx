'use client';

import { useState, useEffect } from 'react';
import { Eye, Shield, Hash, CheckCircle, XCircle, Clock, AlertTriangle, Lock, Zap } from 'lucide-react';
import type { ProveResponse } from '@/lib/types';
import { validatePolicyProof, PolicyValidationResult } from '@/lib/policyRegistry';
import { truncateHash } from '@/lib/contracts';

interface VerifierPaneProps {
  proof?: ProveResponse | null;
  isWaiting?: boolean;
}

export function VerifierPane({ proof, isWaiting }: VerifierPaneProps) {
  const [validation, setValidation] = useState<PolicyValidationResult | null>(null);
  const [verifying, setVerifying] = useState(false);

  useEffect(() => {
    if (proof?.proof?.metadata?.modelHash) {
      verifyProof();
    }
  }, [proof]);

  const verifyProof = async () => {
    if (!proof?.proof) return;

    setVerifying(true);
    // Simulate verification
    await new Promise(resolve => setTimeout(resolve, 500));

    const result = await validatePolicyProof(
      'default-spending-policy',
      proof.proof.metadata.modelHash,
      '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'
    );

    setValidation(result);
    setVerifying(false);
  };

  const decision = proof?.inference?.decision;
  const isApproved = decision === 'approve';

  if (isWaiting || !proof) {
    return (
      <div className="space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            <h3 className="font-semibold">Verifier Public View</h3>
          </div>
          <div className="flex items-center gap-1 text-xs text-cyan-400">
            <Shield className="w-3 h-3" />
            <span>Public only</span>
          </div>
        </div>

        {/* Waiting State */}
        <div className="flex flex-col items-center justify-center py-12 text-gray-500">
          <div className="w-16 h-16 border-2 border-gray-700 border-dashed rounded-full flex items-center justify-center mb-4">
            <Clock className="w-8 h-8 text-gray-600" />
          </div>
          <p className="text-sm">Waiting for proof...</p>
          <p className="text-xs text-gray-600 mt-1">Agent will send proof + public signals</p>
        </div>

        {/* What Verifier Cannot See */}
        <div className="bg-[#0a0a0a] border border-red-500/20 rounded-lg p-4">
          <h4 className="text-xs text-red-400 uppercase tracking-wide mb-2 flex items-center gap-1">
            <Lock className="w-3 h-3" />
            Never Revealed to Verifier
          </h4>
          <ul className="text-xs text-gray-500 space-y-1">
            <li>- Policy thresholds ($X daily limit)</li>
            <li>- Actual budget remaining</li>
            <li>- Behavioral patterns and history</li>
            <li>- Model weights and decision logic</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-cyan-400" />
          <h3 className="font-semibold">Verifier Public View</h3>
        </div>
        <div className="flex items-center gap-1 text-xs text-cyan-400">
          <Shield className="w-3 h-3" />
          <span>Public only</span>
        </div>
      </div>

      {/* Decision */}
      <div className={`p-4 rounded-lg flex items-center gap-4 ${
        isApproved
          ? 'bg-green-500/10 border border-green-500/30'
          : 'bg-red-500/10 border border-red-500/30'
      }`}>
        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
          isApproved ? 'bg-green-500/20' : 'bg-red-500/20'
        }`}>
          {isApproved ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <XCircle className="w-6 h-6 text-red-400" />
          )}
        </div>
        <div>
          <div className={`text-lg font-semibold ${isApproved ? 'text-green-400' : 'text-red-400'}`}>
            {isApproved ? 'APPROVED' : 'REJECTED'}
          </div>
          <div className="text-xs text-gray-400">
            Confidence: {proof.inference?.confidence?.toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Public Signals */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wide flex items-center gap-1">
          <Hash className="w-3 h-3" />
          Public Signals (Visible to Verifier)
        </h4>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-500">policyId</span>
            <span className="font-mono text-purple-400">default-spending-policy</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-500">modelHash</span>
            <span className="font-mono text-white">
              {truncateHash(proof.proof?.metadata.modelHash || '')}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-500">inputHash</span>
            <span className="font-mono text-white">
              {truncateHash(proof.proof?.metadata.inputHash || '')}
            </span>
          </div>
          {proof.proof?.metadata.txIntentHash && (
            <div className="flex justify-between py-2 border-b border-gray-800">
              <span className="text-gray-500">txIntentHash</span>
              <span className="font-mono text-cyan-400">
                {truncateHash(proof.proof.metadata.txIntentHash)}
              </span>
            </div>
          )}
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-500">proofHash</span>
            <span className="font-mono text-white">
              {truncateHash(proof.proof?.proofHash || '')}
            </span>
          </div>
        </div>
      </div>

      {/* Policy Verification */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wide flex items-center gap-1">
          <Shield className="w-3 h-3" />
          Policy Registry Check
        </h4>

        {verifying ? (
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            Verifying against registry...
          </div>
        ) : validation ? (
          <div className={`p-3 rounded-lg border ${
            validation.valid
              ? 'bg-green-500/10 border-green-500/30'
              : 'bg-red-500/10 border-red-500/30'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {validation.valid ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-red-400" />
              )}
              <span className={`text-sm font-medium ${
                validation.valid ? 'text-green-400' : 'text-red-400'
              }`}>
                {validation.valid ? 'Known & Active Policy' : 'Verification Failed'}
              </span>
            </div>
            {!validation.valid && validation.reason && (
              <p className="text-xs text-red-400">{validation.reason}</p>
            )}
          </div>
        ) : null}
      </div>

      {/* Verification Time */}
      <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-800">
        <span className="flex items-center gap-1">
          <Zap className="w-3 h-3 text-green-400" />
          Verification time
        </span>
        <span className="font-mono text-green-400">45ms</span>
      </div>
    </div>
  );
}
