'use client';

import { useState } from 'react';
import { ArrowRight, Check, Globe, Lock, Wallet, RefreshCw, Shield, Zap, DollarSign, Key, CheckCircle, Clock, AlertTriangle } from 'lucide-react';

// CCTP Transfer State
interface CCTPTransferState {
  status: 'idle' | 'proving' | 'burning' | 'attesting' | 'minting' | 'complete' | 'error';
  step: number;
  sourceChain: string;
  destChain: string;
  amount: number;
  proofHash?: string;
  burnTxHash?: string;
  attestation?: string;
  mintTxHash?: string;
  error?: string;
}

// Programmable Wallet State
interface WalletState {
  status: 'idle' | 'creating' | 'configuring' | 'ready';
  walletId?: string;
  address?: string;
  policyId?: string;
  spendingLimit?: number;
  approvedRecipients?: string[];
}

export function CCTPIntegrationDemo() {
  const [transfer, setTransfer] = useState<CCTPTransferState>({
    status: 'idle',
    step: 0,
    sourceChain: 'Ethereum',
    destChain: 'Arc',
    amount: 100,
  });

  const chains = ['Ethereum', 'Arc', 'Avalanche', 'Arbitrum', 'Base', 'Polygon'];

  const startTransfer = async () => {
    // Step 1: Generate spending proof
    setTransfer(prev => ({ ...prev, status: 'proving', step: 1 }));
    await simulateDelay(2000);
    const proofHash = '0x' + Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    setTransfer(prev => ({ ...prev, proofHash }));

    // Step 2: Burn USDC on source chain
    setTransfer(prev => ({ ...prev, status: 'burning', step: 2 }));
    await simulateDelay(1500);
    const burnTxHash = '0x' + Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    setTransfer(prev => ({ ...prev, burnTxHash }));

    // Step 3: Wait for Circle attestation
    setTransfer(prev => ({ ...prev, status: 'attesting', step: 3 }));
    await simulateDelay(3000);
    const attestation = 'cctp_' + Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    setTransfer(prev => ({ ...prev, attestation }));

    // Step 4: Mint on destination with proof
    setTransfer(prev => ({ ...prev, status: 'minting', step: 4 }));
    await simulateDelay(1500);
    const mintTxHash = '0x' + Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
    setTransfer(prev => ({ ...prev, mintTxHash, status: 'complete', step: 5 }));
  };

  const reset = () => {
    setTransfer({
      status: 'idle',
      step: 0,
      sourceChain: 'Ethereum',
      destChain: 'Arc',
      amount: 100,
    });
  };

  return (
    <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center">
          <RefreshCw className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h3 className="font-semibold text-lg">CCTP Cross-Chain Transfer</h3>
          <p className="text-sm text-gray-400">Proof-gated cross-chain USDC transfers via Circle&apos;s CCTP</p>
        </div>
      </div>

      {/* Chain Selection */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-xs text-gray-500 mb-1 block">Source Chain</label>
          <select
            value={transfer.sourceChain}
            onChange={(e) => setTransfer(prev => ({ ...prev, sourceChain: e.target.value }))}
            disabled={transfer.status !== 'idle'}
            className="w-full bg-[#0d1117] border border-gray-700 rounded-lg px-3 py-2 text-sm"
          >
            {chains.filter(c => c !== transfer.destChain).map(chain => (
              <option key={chain} value={chain}>{chain}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">Destination Chain</label>
          <select
            value={transfer.destChain}
            onChange={(e) => setTransfer(prev => ({ ...prev, destChain: e.target.value }))}
            disabled={transfer.status !== 'idle'}
            className="w-full bg-[#0d1117] border border-gray-700 rounded-lg px-3 py-2 text-sm"
          >
            {chains.filter(c => c !== transfer.sourceChain).map(chain => (
              <option key={chain} value={chain}>{chain}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Amount */}
      <div className="mb-6">
        <label className="text-xs text-gray-500 mb-1 block">Amount (USDC)</label>
        <input
          type="number"
          value={transfer.amount}
          onChange={(e) => setTransfer(prev => ({ ...prev, amount: parseFloat(e.target.value) || 0 }))}
          disabled={transfer.status !== 'idle'}
          className="w-full bg-[#0d1117] border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono"
        />
      </div>

      {/* Progress Steps */}
      <div className="space-y-3 mb-6">
        <TransferStep
          step={1}
          title="Generate Spending Proof"
          description="zkML proof that agent follows cross-chain policy"
          status={getStepStatus(transfer.step, 1, transfer.status)}
          hash={transfer.proofHash}
        />
        <TransferStep
          step={2}
          title="Burn USDC on Source"
          description={`Burn ${transfer.amount} USDC on ${transfer.sourceChain}`}
          status={getStepStatus(transfer.step, 2, transfer.status)}
          hash={transfer.burnTxHash}
        />
        <TransferStep
          step={3}
          title="Circle Attestation"
          description="Wait for Circle to attest the burn transaction"
          status={getStepStatus(transfer.step, 3, transfer.status)}
          hash={transfer.attestation}
        />
        <TransferStep
          step={4}
          title="Mint with Proof"
          description={`Mint ${transfer.amount} USDC on ${transfer.destChain} (proof-gated)`}
          status={getStepStatus(transfer.step, 4, transfer.status)}
          hash={transfer.mintTxHash}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        {transfer.status === 'idle' && (
          <button
            onClick={startTransfer}
            className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Start Cross-Chain Transfer
          </button>
        )}
        {transfer.status === 'complete' && (
          <button
            onClick={reset}
            className="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Reset Demo
          </button>
        )}
        {transfer.status !== 'idle' && transfer.status !== 'complete' && (
          <div className="flex-1 bg-blue-500/10 border border-blue-500/30 px-4 py-2 rounded-lg text-sm text-blue-400 text-center">
            Processing...
          </div>
        )}
      </div>

      {/* Code Example */}
      <div className="mt-6 bg-[#0d1117] border border-gray-800 rounded-lg p-4">
        <div className="text-xs text-gray-500 mb-2">Integration Code</div>
        <pre className="text-xs font-mono text-gray-400 overflow-x-auto">
{`import { PolicyProofs } from '@hshadab/spending-proofs';
import { CCTP } from '@circle-fin/cctp-sdk';

// 1. Generate spending proof for cross-chain transfer
const proof = await policyProofs.prove({
  action: 'cross-chain-transfer',
  amount: ${transfer.amount},
  sourceChain: '${transfer.sourceChain}',
  destChain: '${transfer.destChain}',
});

// 2. Execute CCTP transfer with proof attestation
const { burnTx, attestation } = await cctp.depositForBurn({
  amount: ${transfer.amount},
  destinationDomain: getDomain('${transfer.destChain}'),
  mintRecipient: agentAddress,
  proofAttestation: proof.hash, // Proof bound to transfer
});

// 3. Complete on destination (proof verified on-chain)
await cctp.receiveMessage(attestation, { proofHash: proof.hash });`}
        </pre>
      </div>
    </div>
  );
}

export function ProgrammableWalletsDemo() {
  const [wallet, setWallet] = useState<WalletState>({
    status: 'idle',
  });

  const [spendingLimit, setSpendingLimit] = useState(100);
  const [recipients, setRecipients] = useState(['0x742d35Cc6634C0532925a3b844Bc9e7595f8']);

  const createWallet = async () => {
    // Step 1: Create wallet
    setWallet({ status: 'creating' });
    await simulateDelay(1500);
    const walletId = 'wallet_' + Math.random().toString(36).substring(7);
    const address = '0x' + Array.from({ length: 40 }, () => Math.floor(Math.random() * 16).toString(16)).join('');

    // Step 2: Configure policy
    setWallet(prev => ({ ...prev, status: 'configuring', walletId, address }));
    await simulateDelay(1500);
    const policyId = 'policy_' + Math.random().toString(36).substring(7);

    // Step 3: Ready
    setWallet({
      status: 'ready',
      walletId,
      address,
      policyId,
      spendingLimit,
      approvedRecipients: recipients,
    });
  };

  const reset = () => {
    setWallet({ status: 'idle' });
  };

  return (
    <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
          <Wallet className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="font-semibold text-lg">Programmable Wallets</h3>
          <p className="text-sm text-gray-400">Policy-enforced agent wallets via Circle&apos;s Programmable Wallets</p>
        </div>
      </div>

      {wallet.status === 'idle' && (
        <>
          {/* Policy Configuration */}
          <div className="space-y-4 mb-6">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Daily Spending Limit (USDC)</label>
              <input
                type="number"
                value={spendingLimit}
                onChange={(e) => setSpendingLimit(parseFloat(e.target.value) || 0)}
                className="w-full bg-[#0d1117] border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Approved Recipients</label>
              <div className="space-y-2">
                {recipients.map((r, i) => (
                  <div key={i} className="flex gap-2">
                    <input
                      type="text"
                      value={r}
                      onChange={(e) => {
                        const newRecipients = [...recipients];
                        newRecipients[i] = e.target.value;
                        setRecipients(newRecipients);
                      }}
                      className="flex-1 bg-[#0d1117] border border-gray-700 rounded-lg px-3 py-2 text-xs font-mono"
                    />
                    <button
                      onClick={() => setRecipients(recipients.filter((_, j) => j !== i))}
                      className="px-3 py-2 bg-red-500/10 text-red-400 rounded-lg text-xs"
                    >
                      Remove
                    </button>
                  </div>
                ))}
                <button
                  onClick={() => setRecipients([...recipients, ''])}
                  className="w-full py-2 border border-dashed border-gray-700 rounded-lg text-xs text-gray-500 hover:border-gray-600"
                >
                  + Add Recipient
                </button>
              </div>
            </div>
          </div>

          <button
            onClick={createWallet}
            className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            <Key className="w-4 h-4" />
            Create Agent Wallet
          </button>
        </>
      )}

      {(wallet.status === 'creating' || wallet.status === 'configuring') && (
        <div className="space-y-4">
          <div className={`flex items-center gap-3 p-3 rounded-lg ${wallet.status === 'creating' ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-green-500/10 border border-green-500/30'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${wallet.status === 'creating' ? 'bg-purple-500/20' : 'bg-green-500/20'}`}>
              {wallet.status === 'creating' ? (
                <div className="w-3 h-3 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Check className="w-3 h-3 text-green-400" />
              )}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium">Creating Wallet</div>
              {wallet.walletId && <div className="text-xs text-gray-500 font-mono">{wallet.walletId}</div>}
            </div>
          </div>

          <div className={`flex items-center gap-3 p-3 rounded-lg ${wallet.status === 'configuring' ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-gray-800/50 border border-gray-700'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${wallet.status === 'configuring' ? 'bg-purple-500/20' : 'bg-gray-700'}`}>
              {wallet.status === 'configuring' ? (
                <div className="w-3 h-3 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <span className="text-xs text-gray-500">2</span>
              )}
            </div>
            <div className="text-sm text-gray-400">Configuring Policy</div>
          </div>
        </div>
      )}

      {wallet.status === 'ready' && (
        <>
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 mb-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="font-medium text-green-400">Wallet Created</span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Wallet ID</span>
                <span className="font-mono text-gray-300">{wallet.walletId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Address</span>
                <span className="font-mono text-gray-300 text-xs">{wallet.address?.slice(0, 10)}...{wallet.address?.slice(-8)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Policy ID</span>
                <span className="font-mono text-gray-300">{wallet.policyId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Daily Limit</span>
                <span className="font-mono text-green-400">${wallet.spendingLimit} USDC</span>
              </div>
            </div>
          </div>

          <button
            onClick={reset}
            className="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Reset Demo
          </button>
        </>
      )}

      {/* Code Example */}
      <div className="mt-6 bg-[#0d1117] border border-gray-800 rounded-lg p-4">
        <div className="text-xs text-gray-500 mb-2">Integration Code</div>
        <pre className="text-xs font-mono text-gray-400 overflow-x-auto">
{`import { CircleWallets } from '@circle-fin/w3s-sdk';
import { PolicyProofs } from '@hshadab/spending-proofs';

// 1. Create agent wallet with policy enforcement
const wallet = await circleWallets.createWallet({
  idempotencyKey: agentId,
  walletSetId: 'agent-wallets',
  metadata: { agentId, purpose: 'agentic-commerce' },
});

// 2. Configure spending policy
await policyProofs.registerPolicy({
  walletId: wallet.id,
  dailyLimit: ${spendingLimit},
  approvedRecipients: [${recipients.map(r => `'${r}'`).join(', ')}],
  requireProof: true,
});

// 3. Execute proof-gated transfer
const proof = await policyProofs.prove({ /* spending inputs */ });
await circleWallets.createTransaction({
  walletId: wallet.id,
  tokenAddress: USDC_ADDRESS,
  destinationAddress: recipientAddress,
  amount: '10.00',
  proofAttestation: proof.hash, // Required for execution
});`}
        </pre>
      </div>
    </div>
  );
}

// Helper Components
function TransferStep({ step, title, description, status, hash }: {
  step: number;
  title: string;
  description: string;
  status: 'pending' | 'active' | 'complete';
  hash?: string;
}) {
  return (
    <div className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
      status === 'active' ? 'bg-blue-500/10 border border-blue-500/30' :
      status === 'complete' ? 'bg-green-500/10 border border-green-500/30' :
      'bg-gray-800/50 border border-gray-700'
    }`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
        status === 'active' ? 'bg-blue-500/20' :
        status === 'complete' ? 'bg-green-500/20' :
        'bg-gray-700'
      }`}>
        {status === 'active' ? (
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        ) : status === 'complete' ? (
          <Check className="w-4 h-4 text-green-400" />
        ) : (
          <span className="text-sm text-gray-500">{step}</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${
          status === 'active' ? 'text-blue-400' :
          status === 'complete' ? 'text-green-400' :
          'text-gray-400'
        }`}>
          {title}
        </div>
        <div className="text-xs text-gray-500 truncate">{description}</div>
        {hash && (
          <div className="text-xs font-mono text-gray-600 truncate mt-1">
            {hash.slice(0, 20)}...
          </div>
        )}
      </div>
    </div>
  );
}

function getStepStatus(currentStep: number, step: number, status: string): 'pending' | 'active' | 'complete' {
  if (currentStep > step) return 'complete';
  if (currentStep === step && status !== 'idle' && status !== 'complete') return 'active';
  return 'pending';
}

function simulateDelay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Combined Demo Component
export function CircleIntegrationsDemo() {
  return (
    <div className="space-y-8">
      {/* Feature Cards */}
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Globe className="w-5 h-5 text-blue-400" />
            <span className="font-medium">CCTP</span>
          </div>
          <p className="text-sm text-gray-400">
            Cross-chain USDC transfers with proof attestation at mint
          </p>
        </div>
        <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Wallet className="w-5 h-5 text-purple-400" />
            <span className="font-medium">Programmable Wallets</span>
          </div>
          <p className="text-sm text-gray-400">
            Custodial agent wallets with built-in policy enforcement
          </p>
        </div>
        <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-5 h-5 text-green-400" />
            <span className="font-medium">Mint & Redeem</span>
          </div>
          <p className="text-sm text-gray-400">
            Direct USDC mint/redeem with proof-gated compliance
          </p>
        </div>
      </div>

      {/* Interactive Demos */}
      <div className="grid lg:grid-cols-2 gap-6">
        <CCTPIntegrationDemo />
        <ProgrammableWalletsDemo />
      </div>

      {/* Architecture Diagram */}
      <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
        <h3 className="font-semibold mb-4 text-center">Integration Architecture</h3>
        <div className="flex flex-col md:flex-row items-center justify-center gap-4">
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 text-center">
            <div className="text-purple-400 font-medium mb-1">Agent</div>
            <div className="text-xs text-gray-500">Policy Model</div>
          </div>
          <ArrowRight className="w-6 h-6 text-gray-600 rotate-90 md:rotate-0" />
          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 text-center">
            <div className="text-cyan-400 font-medium mb-1">Spending Proofs</div>
            <div className="text-xs text-gray-500">zkML Primitive</div>
          </div>
          <ArrowRight className="w-6 h-6 text-gray-600 rotate-90 md:rotate-0" />
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 text-center">
            <div className="text-blue-400 font-medium mb-1">Circle</div>
            <div className="text-xs text-gray-500">CCTP + Wallets</div>
          </div>
          <ArrowRight className="w-6 h-6 text-gray-600 rotate-90 md:rotate-0" />
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-center">
            <div className="text-green-400 font-medium mb-1">Arc</div>
            <div className="text-xs text-gray-500">Settlement</div>
          </div>
        </div>
      </div>
    </div>
  );
}
