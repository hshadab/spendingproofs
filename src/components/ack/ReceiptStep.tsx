'use client';

import { Receipt, CheckCircle, Loader2, Shield, FileCheck } from 'lucide-react';
import type { ACKPaymentReceipt } from '@/lib/ack/types';
import { getReceiptSummary } from '@/lib/ack/payments';

interface ReceiptStepProps {
  receipt: ACKPaymentReceipt | null;
  isIssuing: boolean;
  onIssueReceipt: () => Promise<void>;
  disabled?: boolean;
}

export function ReceiptStep({
  receipt,
  isIssuing,
  onIssueReceipt,
  disabled,
}: ReceiptStepProps) {
  if (receipt) {
    const summary = getReceiptSummary(receipt);

    return (
      <div className="bg-[#0d1117] border border-green-800/50 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">5. Receipt Issued</h3>
            <p className="text-sm text-green-400">Verifiable credential created</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Credential Type</span>
            <span className="text-purple-400 text-sm">PaymentReceiptCredential</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Amount</span>
            <span className="text-white font-medium">{summary.amount}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Transaction</span>
            <span className="text-cyan-400 font-mono text-sm">
              {summary.txHash.slice(0, 10)}...{summary.txHash.slice(-6)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Proof Hash</span>
            <span className="text-cyan-400 font-mono text-sm">
              {summary.proofHash.slice(0, 10)}...{summary.proofHash.slice(-6)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Network</span>
            <span className="text-white">{summary.network}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Issued At</span>
            <span className="text-white text-sm">{summary.issuedAt}</span>
          </div>
        </div>

        <div className="mt-4 p-3 bg-green-900/20 border border-green-800 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <FileCheck className="w-4 h-4" />
            <span>W3C Verifiable Credential - Audit-ready</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0d1117] border border-purple-800/50 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
          <Receipt className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">5. Issue Receipt</h3>
          <p className="text-sm text-gray-400">Create verifiable payment receipt</p>
        </div>
      </div>

      <div className="mb-4 p-4 bg-gray-900/50 border border-gray-800 rounded-lg">
        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
          <Shield className="w-4 h-4" />
          <span>ACK-Pay Receipt</span>
        </div>
        <p className="text-xs text-gray-500">
          Issues a W3C PaymentReceiptCredential proving the payment was executed
          with a verified spending policy.
        </p>
      </div>

      <button
        onClick={onIssueReceipt}
        disabled={disabled || isIssuing}
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {isIssuing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Issuing Receipt...
          </>
        ) : (
          <>
            <Receipt className="w-5 h-5" />
            Issue Payment Receipt
          </>
        )}
      </button>

      <p className="mt-4 text-xs text-gray-500">
        Creates a verifiable credential linking the transaction to the zkML proof.
      </p>
    </div>
  );
}
