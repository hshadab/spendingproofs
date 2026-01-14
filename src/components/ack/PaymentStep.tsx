'use client';

import { useState } from 'react';
import { CreditCard, CheckCircle, Loader2, AlertCircle, ExternalLink } from 'lucide-react';
import { getExplorerTxUrl } from '@/lib/config';

interface PaymentStepProps {
  txHash: string | null;
  isExecuting: boolean;
  error: string | null;
  onExecutePayment: (recipient: string, amount: string) => Promise<void>;
  disabled?: boolean;
  mode?: 'demo' | 'live';
}

export function PaymentStep({
  txHash,
  isExecuting,
  error,
  onExecutePayment,
  disabled,
  mode = 'demo',
}: PaymentStepProps) {
  const [recipient, setRecipient] = useState('0x8ba1f109551bD432803012645Ac136ddd64DBA72');
  const [amount, setAmount] = useState('0.01');

  const handleExecute = async () => {
    await onExecutePayment(recipient, amount);
  };

  if (txHash) {
    return (
      <div className="bg-[#0d1117] border border-green-800/50 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">4. Payment Executed</h3>
            <p className="text-sm text-green-400">On-chain transfer complete</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Amount</span>
            <span className="text-white font-medium">${amount} USDC</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Recipient</span>
            <span className="text-white font-mono text-sm">
              {recipient.slice(0, 6)}...{recipient.slice(-4)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Network</span>
            <span className="text-cyan-400">Arc Testnet</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Transaction</span>
            <a
              href={getExplorerTxUrl(txHash)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-purple-400 font-mono text-sm flex items-center gap-1 hover:underline"
            >
              {txHash.slice(0, 10)}...{txHash.slice(-6)}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0d1117] border border-purple-800/50 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
          <CreditCard className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">4. Execute Payment</h3>
          <p className="text-sm text-gray-400">Transfer USDC on Arc Testnet</p>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Recipient Address</label>
          <input
            type="text"
            value={recipient}
            onChange={(e) => setRecipient(e.target.value)}
            disabled={disabled || isExecuting}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white font-mono text-sm focus:border-purple-500 focus:outline-none disabled:opacity-50"
            placeholder="0x..."
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Amount (USDC)</label>
          <input
            type="text"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            disabled={disabled || isExecuting}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none disabled:opacity-50"
            placeholder="0.01"
          />
        </div>

        {error && (
          <div className="p-3 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        <button
          onClick={handleExecute}
          disabled={disabled || isExecuting || !recipient || !amount}
          className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {isExecuting ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Executing Payment...
            </>
          ) : (
            <>
              <CreditCard className="w-5 h-5" />
              Execute Payment
            </>
          )}
        </button>
      </div>

      <p className="mt-4 text-xs text-gray-500">
        {mode === 'live'
          ? 'Executes a gated USDC transfer on Arc Testnet using the verified zkML proof.'
          : 'Simulates a gated USDC transfer (Demo Mode - no real transaction).'}
      </p>
    </div>
  );
}
