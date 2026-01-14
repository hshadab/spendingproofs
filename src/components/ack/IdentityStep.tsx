'use client';

import { useState } from 'react';
import { User, CheckCircle, Loader2, Shield, AlertCircle } from 'lucide-react';
import { formatDid } from '@/lib/ack/client';
import type { ACKAgentIdentity } from '@/lib/ack/types';

interface IdentityStepProps {
  identity: ACKAgentIdentity | null;
  isCreating: boolean;
  onCreateIdentity: (name: string) => Promise<void>;
  disabled?: boolean;
  mode?: 'demo' | 'live';
  isWalletConnected?: boolean;
}

export function IdentityStep({
  identity,
  isCreating,
  onCreateIdentity,
  disabled,
  mode = 'demo',
  isWalletConnected = false,
}: IdentityStepProps) {
  const [agentName, setAgentName] = useState('Spending Agent');

  const handleCreate = async () => {
    await onCreateIdentity(agentName);
  };

  // In Live Mode, show warning if wallet not connected
  const needsWallet = mode === 'live' && !isWalletConnected;

  if (identity) {
    return (
      <div className="bg-[#0d1117] border border-green-800/50 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
            <CheckCircle className="w-5 h-5 text-green-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">1. Agent Identity Created</h3>
            <p className="text-sm text-green-400">
              {mode === 'live' ? 'Live identity verified' : 'Demo identity created'}
            </p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Agent Name</span>
            <span className="text-white font-medium">{identity.name}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">DID</span>
            <span className="text-cyan-400 font-mono text-sm">{formatDid(identity.did)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Owner</span>
            <span className="text-white font-mono text-sm">
              {identity.ownerAddress.slice(0, 6)}...{identity.ownerAddress.slice(-4)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Credential Type</span>
            <span className="text-purple-400 text-sm">ControllerCredential</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Mode</span>
            <span className={`text-sm ${mode === 'live' ? 'text-green-400' : 'text-purple-400'}`}>
              {mode === 'live' ? 'Live' : 'Demo'}
            </span>
          </div>
        </div>

        <div className="mt-4 p-3 bg-green-900/20 border border-green-800 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <Shield className="w-4 h-4" />
            <span>
              {mode === 'live'
                ? 'Verifiable identity linked to your wallet'
                : 'Demo identity created (simulated)'}
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0d1117] border border-purple-800/50 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
          <User className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">1. Create Agent Identity</h3>
          <p className="text-sm text-gray-400">Generate ACK-ID for your agent</p>
        </div>
      </div>

      {needsWallet && (
        <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-800 rounded-lg flex items-center gap-2 text-yellow-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          <span>Connect your wallet above to create a live identity</span>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Agent Name</label>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            disabled={disabled || isCreating || needsWallet}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none disabled:opacity-50"
            placeholder="Enter agent name"
          />
        </div>

        <button
          onClick={handleCreate}
          disabled={disabled || isCreating || !agentName.trim() || needsWallet}
          className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {isCreating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Creating Identity...
            </>
          ) : (
            <>
              <User className="w-5 h-5" />
              Create Agent Identity
            </>
          )}
        </button>
      </div>

      <p className="mt-4 text-xs text-gray-500">
        {mode === 'live'
          ? 'Creates a W3C DID linked to your connected wallet.'
          : 'Creates a demo W3C DID with simulated wallet address.'}
      </p>
    </div>
  );
}
