'use client';

import { useState } from 'react';
import { Shield, XCircle, CheckCircle, AlertTriangle, Play, RotateCcw, Lock, Zap, DollarSign, ExternalLink, Wallet } from 'lucide-react';
import { useDemoWallet } from '@/hooks/useDemoWallet';

interface ScenarioResult {
  status: 'idle' | 'running' | 'success' | 'reverted';
  message?: string;
  txHash?: string;
  explorerUrl?: string;
}

interface EnforcementDemoProps {
  proofHash?: string | null;
  attestationTxHash?: string | null;
  onReset?: () => void;
}

const DEMO_MERCHANT = '0x8ba1f109551bD432803012645Ac136ddd64DBA72';
const TRANSFER_AMOUNT = 0.01; // Use 0.01 USDC to make demo wallet last longer

export function EnforcementDemo({ proofHash, attestationTxHash, onReset }: EnforcementDemoProps) {
  const {
    status: walletStatus,
    isLoading,
    executeGatedTransfer,
    address,
    balance,
    spendingGate,
    fetchStatus,
  } = useDemoWallet();

  const [scenarios, setScenarios] = useState<{
    noProof: ScenarioResult;
    validProof: ScenarioResult;
    replayAttack: ScenarioResult;
  }>({
    noProof: { status: 'idle' },
    validProof: { status: 'idle' },
    replayAttack: { status: 'idle' },
  });

  const [usedProofHash, setUsedProofHash] = useState<string | null>(null);

  const hasLowBalance = spendingGate && parseFloat(spendingGate.balance) < TRANSFER_AMOUNT;
  const isConfigured = spendingGate?.configured;

  const runScenario = async (scenario: 'noProof' | 'validProof' | 'replayAttack') => {
    setScenarios(prev => ({
      ...prev,
      [scenario]: { status: 'running' },
    }));

    let result;
    const expiry = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now

    switch (scenario) {
      case 'noProof':
        // Try to transfer with an invalid/fake proof hash
        result = await executeGatedTransfer({
          to: DEMO_MERCHANT,
          amount: TRANSFER_AMOUNT,
          proofHash: '0x0000000000000000000000000000000000000000000000000000000000000000',
          expiry,
        });
        break;

      case 'validProof':
        if (!proofHash) {
          setScenarios(prev => ({
            ...prev,
            [scenario]: {
              status: 'reverted',
              message: 'No proof available. Generate a proof first in the Payment Flow above.'
            },
          }));
          return;
        }
        result = await executeGatedTransfer({
          to: DEMO_MERCHANT,
          amount: TRANSFER_AMOUNT,
          proofHash: proofHash as `0x${string}`,
          expiry,
        });
        if (result.success) {
          setUsedProofHash(proofHash);
        }
        break;

      case 'replayAttack':
        // Try to reuse the same proof hash
        const replayHash = usedProofHash || proofHash;
        if (!replayHash) {
          setScenarios(prev => ({
            ...prev,
            [scenario]: {
              status: 'reverted',
              message: 'Run "Valid Proof" scenario first to have a proof to replay.'
            },
          }));
          return;
        }
        result = await executeGatedTransfer({
          to: DEMO_MERCHANT,
          amount: TRANSFER_AMOUNT,
          proofHash: replayHash as `0x${string}`,
          expiry,
        });
        break;
    }

    setScenarios(prev => ({
      ...prev,
      [scenario]: {
        status: result.success ? 'success' : 'reverted',
        message: result.success ? undefined : result.revertReason || result.error,
        txHash: result.hash,
        explorerUrl: result.explorerUrl,
      },
    }));

    // Refresh balance after transaction
    fetchStatus();
  };

  const resetAll = () => {
    setUsedProofHash(null);
    setScenarios({
      noProof: { status: 'idle' },
      validProof: { status: 'idle' },
      replayAttack: { status: 'idle' },
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
    disabled,
  }: {
    title: string;
    description: string;
    scenarioKey: 'noProof' | 'validProof' | 'replayAttack';
    expectedOutcome: 'success' | 'revert';
    icon: typeof Shield;
    iconColor: string;
    disabled?: boolean;
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
            disabled={isLoading || disabled}
            className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            <Play className="w-4 h-4" />
            Execute On-Chain
          </button>
        )}

        {isRunning && (
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
            <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
            Executing on Arc Testnet...
          </div>
        )}

        {result.status === 'success' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-green-400">
              <CheckCircle className="w-4 h-4" />
              Transaction succeeded
            </div>
            {result.explorerUrl && (
              <a
                href={result.explorerUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-green-400 hover:underline"
              >
                View on ArcScan <ExternalLink className="w-3 h-3" />
              </a>
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
            Real On-Chain Enforcement
          </h3>
          <p className="text-sm text-gray-400">
            Execute real transactions on Arc Testnet. Without valid proofs, transactions <span className="text-red-400 font-medium">REVERT</span>.
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

      {/* Wallet Status */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="text-sm font-medium text-white">Demo Wallet</div>
              <div className="text-xs text-gray-400 font-mono">
                {address ? `${address.slice(0, 6)}...${address.slice(-4)}` : 'Loading...'}
              </div>
            </div>
          </div>
          <div className="text-right">
            {balance && (
              <div className="text-sm">
                <span className="text-gray-400">Wallet: </span>
                <span className="text-green-400 font-mono">{balance.usdc} USDC</span>
              </div>
            )}
            {spendingGate && (
              <div className="text-sm">
                <span className="text-gray-400">SpendingGate: </span>
                <span className="text-cyan-400 font-mono">{spendingGate.balance} USDC</span>
              </div>
            )}
          </div>
        </div>

        {hasLowBalance && (
          <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-amber-400">
                <strong>Low SpendingGate Balance</strong>
                <p className="text-amber-400/70 mt-1">
                  The SpendingGate wallet needs more USDC to demo enforcement.
                  Deposit USDC to the SpendingGate contract at{' '}
                  <span className="font-mono">{spendingGate?.address?.slice(0, 10)}...</span>
                </p>
              </div>
            </div>
          </div>
        )}

        {!isConfigured && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-red-400">
                <strong>SpendingGate Not Configured</strong>
                <p className="text-red-400/70 mt-1">
                  The SpendingGate contract is not deployed or configured.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Current Proof Info */}
      {proofHash && (
        <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
          <h4 className="text-sm font-semibold mb-3 text-gray-300">Current Proof</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-gray-500">Proof Hash</span>
              <div className="font-mono text-cyan-400 break-all">{proofHash.slice(0, 20)}...{proofHash.slice(-8)}</div>
            </div>
            <div>
              <span className="text-gray-500">Transfer Amount</span>
              <div className="font-mono text-white">{TRANSFER_AMOUNT} USDC</div>
            </div>
          </div>
          {attestationTxHash && (
            <div className="mt-3">
              <a
                href={`https://testnet.arcscan.app/tx/${attestationTxHash}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-purple-400 hover:underline flex items-center gap-1"
              >
                View Attestation TX <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}
        </div>
      )}

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
          Real on-chain enforcement: SpendingGate reverts without attested proof
        </p>
      </div>

      {/* Scenarios */}
      <div className="grid md:grid-cols-3 gap-4">
        <ScenarioCard
          title="No Valid Proof"
          description="Attempt transfer with fake/unattested proof hash. Contract checks attestation and reverts."
          scenarioKey="noProof"
          expectedOutcome="revert"
          icon={XCircle}
          iconColor="bg-red-500/10 text-red-400"
          disabled={hasLowBalance || !isConfigured}
        />

        <ScenarioCard
          title="Valid Attested Proof"
          description="Transfer with real attested proof. Contract verifies and executes USDC transfer."
          scenarioKey="validProof"
          expectedOutcome="success"
          icon={CheckCircle}
          iconColor="bg-green-500/10 text-green-400"
          disabled={hasLowBalance || !isConfigured || !proofHash}
        />

        <ScenarioCard
          title="Replay Attack"
          description="Try to reuse an already-consumed proof. Contract detects replay and reverts."
          scenarioKey="replayAttack"
          expectedOutcome="revert"
          icon={Shield}
          iconColor="bg-purple-500/10 text-purple-400"
          disabled={hasLowBalance || !isConfigured || !usedProofHash}
        />
      </div>

      {/* Key Takeaways */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
        <h4 className="text-sm font-semibold mb-3">Key Takeaways</h4>
        <ul className="space-y-2 text-xs text-gray-400">
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>These are <strong className="text-white">real on-chain transactions</strong> on Arc Testnet</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>SpendingGate <strong className="text-white">enforces</strong> proof requirements - transfers revert without valid proofs</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Proof must be attested on-chain before it can be used for gated transfers</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
            <span>Each proof can only be used <strong className="text-white">once</strong> - replay attacks are blocked</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
