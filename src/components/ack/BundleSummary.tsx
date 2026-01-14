'use client';

import { Package, CheckCircle, ExternalLink, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import type { ACKAgentIdentity, ACKPaymentReceipt } from '@/lib/ack/types';
import type { SpendingProof } from '@/lib/types';
import { formatDid } from '@/lib/ack/client';
import { getExplorerTxUrl } from '@/lib/config';

interface BundleSummaryProps {
  identity: ACKAgentIdentity | null;
  proof: SpendingProof | null;
  receipt: ACKPaymentReceipt | null;
}

export function BundleSummary({ identity, proof, receipt }: BundleSummaryProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const isComplete = identity && proof && receipt;

  const copyToClipboard = async (text: string, field: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  return (
    <div className={`bg-[#0d1117] border rounded-xl p-6 ${
      isComplete ? 'border-green-800/50' : 'border-gray-800'
    }`}>
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
          isComplete ? 'bg-green-500/20' : 'bg-gray-800'
        }`}>
          {isComplete ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <Package className="w-5 h-5 text-gray-400" />
          )}
        </div>
        <div>
          <h3 className="font-semibold text-white">ACK Proof Bundle</h3>
          <p className={`text-sm ${isComplete ? 'text-green-400' : 'text-gray-400'}`}>
            {isComplete ? 'Complete - All artifacts generated' : 'In progress...'}
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {/* Identity */}
        <div className={`p-4 rounded-lg border ${
          identity ? 'bg-green-900/10 border-green-800/30' : 'bg-gray-900/50 border-gray-800'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">ACK-ID</span>
            {identity && (
              <span className="text-xs text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Verified
              </span>
            )}
          </div>
          {identity ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">DID</span>
                <div className="flex items-center gap-2">
                  <span className="text-cyan-400 font-mono text-xs">{formatDid(identity.did, 20)}</span>
                  <button
                    onClick={() => copyToClipboard(identity.did, 'did')}
                    className="text-gray-500 hover:text-white"
                  >
                    {copiedField === 'did' ? (
                      <Check className="w-3 h-3 text-green-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </button>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Agent</span>
                <span className="text-white text-xs">{identity.name}</span>
              </div>
            </div>
          ) : (
            <span className="text-gray-500 text-xs">Pending...</span>
          )}
        </div>

        {/* Proof */}
        <div className={`p-4 rounded-lg border ${
          proof ? 'bg-green-900/10 border-green-800/30' : 'bg-gray-900/50 border-gray-800'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">zkML Proof</span>
            {proof && (
              <span className="text-xs text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Generated
              </span>
            )}
          </div>
          {proof ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Hash</span>
                <div className="flex items-center gap-2">
                  <span className="text-purple-400 font-mono text-xs">
                    {proof.proofHash.slice(0, 14)}...
                  </span>
                  <button
                    onClick={() => copyToClipboard(proof.proofHash, 'proof')}
                    className="text-gray-500 hover:text-white"
                  >
                    {copiedField === 'proof' ? (
                      <Check className="w-3 h-3 text-green-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </button>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Decision</span>
                <span className={`text-xs ${proof.decision.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
                  {proof.decision.shouldBuy ? 'APPROVED' : 'REJECTED'}
                </span>
              </div>
            </div>
          ) : (
            <span className="text-gray-500 text-xs">Pending...</span>
          )}
        </div>

        {/* Receipt */}
        <div className={`p-4 rounded-lg border ${
          receipt ? 'bg-green-900/10 border-green-800/30' : 'bg-gray-900/50 border-gray-800'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">ACK-Pay Receipt</span>
            {receipt && (
              <span className="text-xs text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Issued
              </span>
            )}
          </div>
          {receipt ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Transaction</span>
                <a
                  href={getExplorerTxUrl(receipt.txHash)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan-400 font-mono text-xs flex items-center gap-1 hover:underline"
                >
                  {receipt.txHash.slice(0, 10)}...
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Amount</span>
                <span className="text-white text-xs">${receipt.amount} USDC</span>
              </div>
            </div>
          ) : (
            <span className="text-gray-500 text-xs">Pending...</span>
          )}
        </div>
      </div>

      {isComplete && (
        <div className="mt-6 p-4 bg-green-900/20 border border-green-800 rounded-lg">
          <p className="text-green-400 text-sm text-center">
            Complete audit trail: Identity + Policy Proof + Payment Receipt
          </p>
        </div>
      )}
    </div>
  );
}
