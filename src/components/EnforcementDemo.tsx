'use client';

import { useState } from 'react';
import { Shield, XCircle, CheckCircle, AlertTriangle, Play, RotateCcw, Lock, Zap, DollarSign } from 'lucide-react';
import { gatedTransfer, TxIntent, resetContractState, computeTxIntentHash } from '@/lib/spendingGate';
import { SpendingProof } from '@/lib/types';

interface ScenarioResult {
  status: 'idle' | 'running' | 'success' | 'reverted';
  message?: string;
  txHash?: string;
}

interface EnforcementDemoProps {
  proof?: SpendingProof | null;
  onReset?: () => void;
}

const mockProof: SpendingProof = {
  proofHash: '0x7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b',
  inputHash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
  modelHash: '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
  decision: { shouldBuy: true, confidence: 87, riskScore: 12 },
  timestamp: Date.now(),
  proofSizeBytes: 48500,
  generationTimeMs: 2100,
  verified: true,
};

const baseTxIntent: TxIntent = {
  chainId: 5042002,
  usdcAddress: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
  sender: '0x742d35Cc6634C0532925a3b844Bc9e7595f8fE32',
  recipient: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',
  amount: BigInt(50000), // 0.05 USDC (6 decimals)
  nonce: BigInt(1),
  expiry: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
  policyId: 'default-spending-policy',
  policyVersion: 1,
};

export function EnforcementDemo({ proof, onReset }: EnforcementDemoProps) {
  const [scenarios, setScenarios] = useState<{
    noProof: ScenarioResult;
    validProof: ScenarioResult;
    modifiedAmount: ScenarioResult;
    replay: ScenarioResult;
  }>({
    noProof: { status: 'idle' },
    validProof: { status: 'idle' },
    modifiedAmount: { status: 'idle' },
    replay: { status: 'idle' },
  });

  const [hasRunValidProof, setHasRunValidProof] = useState(false);

  const runScenario = async (
    scenario: 'noProof' | 'validProof' | 'modifiedAmount' | 'replay'
  ) => {
    setScenarios(prev => ({
      ...prev,
      [scenario]: { status: 'running' },
    }));

    const activeProof = proof || mockProof;
    let result;

    switch (scenario) {
      case 'noProof':
        result = await gatedTransfer(baseTxIntent, null, { skipProof: true });
        break;
      case 'validProof':
        result = await gatedTransfer(
          { ...baseTxIntent, nonce: BigInt(Date.now()) }, // Fresh nonce
          activeProof
        );
        if (result.success) setHasRunValidProof(true);
        break;
      case 'modifiedAmount':
        result = await gatedTransfer(baseTxIntent, activeProof, { modifyAmount: true });
        break;
      case 'replay':
        result = await gatedTransfer(baseTxIntent, activeProof, { replayNonce: true });
        break;
    }

    setScenarios(prev => ({
      ...prev,
      [scenario]: {
        status: result.success ? 'success' : 'reverted',
        message: result.success ? undefined : result.revertReason,
        txHash: result.txHash,
      },
    }));
  };

  const resetAll = () => {
    resetContractState();
    setHasRunValidProof(false);
    setScenarios({
      noProof: { status: 'idle' },
      validProof: { status: 'idle' },
      modifiedAmount: { status: 'idle' },
      replay: { status: 'idle' },
    });
    onReset?.();
  };

  const ScenarioCard = ({
    title,
    description,
    scenarioKey,
    expectedOutcome,
    icon: Icon,
    iconColor,
  }: {
    title: string;
    description: string;
    scenarioKey: 'noProof' | 'validProof' | 'modifiedAmount' | 'replay';
    expectedOutcome: 'success' | 'revert';
    icon: typeof Shield;
    iconColor: string;
  }) => {
    const result = scenarios[scenarioKey];
    const isRunning = result.status === 'running';

    return (
      <div className={`bg-[#0d1117] border rounded-xl p-5 transition-all ${
        result.status === 'success'
          ? 'border-green-500/50'
          : result.status === 'reverted'
          ? 'border-red-500/50'
          : 'border-gray-800'
      }`}>
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${iconColor}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-sm">{title}</h4>
              <p className="text-xs text-gray-500">
                Expected: <span className={expectedOutcome === 'success' ? 'text-green-400' : 'text-red-400'}>
                  {expectedOutcome === 'success' ? 'SUCCESS' : 'REVERT'}
                </span>
              </p>
            </div>
          </div>
          {result.status === 'success' && (
            <CheckCircle className="w-5 h-5 text-green-400" />
          )}
          {result.status === 'reverted' && (
            <XCircle className="w-5 h-5 text-red-400" />
          )}
        </div>

        <p className="text-xs text-gray-400 mb-4">{description}</p>

        {result.status === 'idle' && (
          <button
            onClick={() => runScenario(scenarioKey)}
            className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            <Play className="w-4 h-4" />
            Execute
          </button>
        )}

        {isRunning && (
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
            <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
            Executing transaction...
          </div>
        )}

        {result.status === 'success' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-green-400">
              <CheckCircle className="w-4 h-4" />
              Transaction succeeded
            </div>
            {result.txHash && (
              <div className="text-xs font-mono text-gray-500 break-all">
                tx: {result.txHash.slice(0, 20)}...
              </div>
            )}
          </div>
        )}

        {result.status === 'reverted' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-red-400">
              <XCircle className="w-4 h-4" />
              Transaction reverted
            </div>
            <div className="text-xs text-gray-400 bg-red-500/10 p-2 rounded border border-red-500/20">
              {result.message}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-xl font-bold mb-2 flex items-center gap-2">
            <Lock className="w-5 h-5 text-purple-400" />
            Enforcement Demo
          </h3>
          <p className="text-sm text-gray-400">
            See how SpendingGate enforces policy compliance. Without valid proofs, transactions <span className="text-red-400 font-medium">REVERT</span>.
          </p>
        </div>
        <button
          onClick={resetAll}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Current State */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
        <h4 className="text-sm font-semibold mb-3 text-gray-300">Transaction Intent</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div>
            <span className="text-gray-500">Amount</span>
            <div className="font-mono text-white">0.05 USDC</div>
          </div>
          <div>
            <span className="text-gray-500">Recipient</span>
            <div className="font-mono text-white">{baseTxIntent.recipient.slice(0, 10)}...</div>
          </div>
          <div>
            <span className="text-gray-500">Policy</span>
            <div className="font-mono text-purple-400">{baseTxIntent.policyId}</div>
          </div>
          <div>
            <span className="text-gray-500">txIntentHash</span>
            <div className="font-mono text-cyan-400">{computeTxIntentHash(baseTxIntent).slice(0, 10)}...</div>
          </div>
        </div>
      </div>

      {/* Flow Diagram */}
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-500/20 rounded-xl p-4">
        <div className="flex items-center justify-between gap-4 text-xs overflow-x-auto">
          <div className="flex flex-col items-center gap-1 min-w-[80px]">
            <div className="w-8 h-8 bg-purple-500/20 rounded-full flex items-center justify-center">
              <DollarSign className="w-4 h-4 text-purple-400" />
            </div>
            <span className="text-gray-400">Agent</span>
          </div>
          <div className="flex-1 h-0.5 bg-gradient-to-r from-purple-500 to-cyan-500" />
          <div className="flex flex-col items-center gap-1 min-w-[80px]">
            <div className="w-8 h-8 bg-cyan-500/20 rounded-full flex items-center justify-center">
              <Shield className="w-4 h-4 text-cyan-400" />
            </div>
            <span className="text-gray-400">Proof Check</span>
          </div>
          <div className="flex-1 h-0.5 bg-gradient-to-r from-cyan-500 to-green-500" />
          <div className="flex flex-col items-center gap-1 min-w-[80px]">
            <div className="w-8 h-8 bg-amber-500/20 rounded-full flex items-center justify-center">
              <Lock className="w-4 h-4 text-amber-400" />
            </div>
            <span className="text-gray-400">SpendingGate</span>
          </div>
          <div className="flex-1 h-0.5 bg-gradient-to-r from-amber-500 to-green-500" />
          <div className="flex flex-col items-center gap-1 min-w-[80px]">
            <div className="w-8 h-8 bg-green-500/20 rounded-full flex items-center justify-center">
              <Zap className="w-4 h-4 text-green-400" />
            </div>
            <span className="text-gray-400">Transfer</span>
          </div>
        </div>
        <p className="text-xs text-gray-500 text-center mt-3">
          SpendingGate contract requires valid proof before executing USDC transfer
        </p>
      </div>

      {/* Scenarios */}
      <div className="grid md:grid-cols-2 gap-4">
        <ScenarioCard
          title="No Proof Provided"
          description="Agent attempts to transfer USDC without submitting any spending proof."
          scenarioKey="noProof"
          expectedOutcome="revert"
          icon={XCircle}
          iconColor="bg-red-500/10 text-red-400"
        />

        <ScenarioCard
          title="Valid Proof"
          description="Agent submits valid spending proof with matching txIntentHash. Transfer succeeds."
          scenarioKey="validProof"
          expectedOutcome="success"
          icon={CheckCircle}
          iconColor="bg-green-500/10 text-green-400"
        />

        <ScenarioCard
          title="Modified Amount"
          description="Agent tries to transfer different amount than what was proven. Intent hash mismatch."
          scenarioKey="modifiedAmount"
          expectedOutcome="revert"
          icon={AlertTriangle}
          iconColor="bg-amber-500/10 text-amber-400"
        />

        <ScenarioCard
          title="Replay Attack"
          description="Agent tries to reuse the same proof/nonce. Already consumed."
          scenarioKey="replay"
          expectedOutcome="revert"
          icon={Shield}
          iconColor="bg-purple-500/10 text-purple-400"
        />
      </div>

      {/* Status Summary */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
        <h4 className="text-sm font-semibold mb-3">Key Takeaways</h4>
        <ul className="space-y-2 text-xs text-gray-400">
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>SpendingGate <strong className="text-white">enforces</strong> proof requirements - transfers revert without valid proofs</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>txIntentHash binds proofs to specific transaction parameters (amount, recipient, nonce)</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Nonce tracking prevents replay attacks - each proof can only be used once</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>This is <strong className="text-white">hard enforcement</strong>, not just advisory attestation</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
