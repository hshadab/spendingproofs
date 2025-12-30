'use client';

import Link from 'next/link';
import {
  ArrowLeft,
  Cpu,
  CheckCircle,
  Clock,
  DollarSign,
  Shield,
  Zap,
  Server,
  Code,
  ArrowRight,
  XCircle,
  AlertTriangle,
} from 'lucide-react';
import { ArchitectureDiagram } from '@/components/diagrams/ArchitectureDiagram';

const TIMELINE_STEPS = [
  {
    step: 1,
    title: 'Agent Receives Task',
    description: 'Agent X is tasked with training a machine learning model. It determines GPU compute is needed.',
    timing: 't=0ms',
    icon: <Cpu className="w-5 h-5" />,
    code: `// Agent analyzes task requirements
const requirements = await agent.analyzeTask(task);
// Result: { gpuType: "A100", hours: 4, maxPrice: 100 }`,
  },
  {
    step: 2,
    title: 'Query Compute Marketplace',
    description: 'Agent queries available GPU instances from decentralized compute marketplace.',
    timing: 't=50ms',
    icon: <Server className="w-5 h-5" />,
    code: `// Query marketplace for available compute
const offers = await marketplace.getOffers({
  gpuType: "A100",
  minHours: 4,
  maxPricePerHour: 25
});
// Returns: [{ provider: "0x...", price: 22, trust: 0.92 }]`,
  },
  {
    step: 3,
    title: 'Policy Evaluation',
    description: 'Transaction inputs are fed through the spending policy model to determine approval.',
    timing: 't=100ms',
    icon: <Shield className="w-5 h-5" />,
    code: `// Prepare model inputs
const inputs = {
  price: 88,           // $22/hr × 4 hours
  merchantTrust: 0.92, // Provider reputation
  categoryMatch: 0.95, // GPU compute is allowed
  urgency: 0.7,        // Task deadline approaching
  volatility: 0.1,     // Stable pricing
  stockLevel: 0.8,     // Good availability
  promotionalDiscount: 0,
  historicalAccuracy: 0.88
};`,
  },
  {
    step: 4,
    title: 'Proof Generation',
    description: 'Jolt prover generates HyperKZG proof of policy compliance.',
    timing: 't=2300ms',
    icon: <Zap className="w-5 h-5" />,
    code: `// Generate spending proof
const { proof, decision } = await spendingSDK.prove(inputs);
// proof.hash: "0x7a3f..."
// proof.size: 48128 bytes
// decision: "approve"`,
  },
  {
    step: 5,
    title: 'Submit to Arc',
    description: 'Proof is attested on-chain and USDC payment is executed atomically.',
    timing: 't=2400ms',
    icon: <CheckCircle className="w-5 h-5" />,
    code: `// Submit attestation + payment
const tx = await spendingGate.gatedTransfer({
  recipient: provider.address,
  amount: 88_000000n, // 88 USDC
  proofHash: proof.hash,
  txIntentHash: computeTxIntentHash(intent)
});
// Finality: <1 second`,
  },
  {
    step: 6,
    title: 'Compute Provisioned',
    description: 'Provider verifies proof attestation and provisions GPU instance for agent.',
    timing: 't=2500ms',
    icon: <Cpu className="w-5 h-5" />,
    code: `// Provider verifies and provisions
const isValid = await provider.verifyAttestation(tx.hash);
if (isValid) {
  const instance = await provider.provision({
    client: agent.address,
    gpuType: "A100",
    hours: 4
  });
}`,
  },
];

const COMPARISON_TABLE = [
  {
    aspect: 'Gas Token',
    arc: 'USDC (stable)',
    ethereum: 'ETH (volatile)',
    issue: 'ETH price spike could exceed budget mid-transaction',
  },
  {
    aspect: 'Finality',
    arc: '<1 second (deterministic)',
    ethereum: '~12 minutes (probabilistic)',
    issue: 'Agent can\'t proceed with compute until payment confirms',
  },
  {
    aspect: 'Reorg Risk',
    arc: 'None',
    ethereum: 'Possible',
    issue: 'Payment could reverse after compute is provisioned',
  },
  {
    aspect: 'MEV/Front-running',
    arc: 'Opt-in privacy available',
    ethereum: 'Fully public mempool',
    issue: 'Competitors could front-run compute purchases',
  },
];

const METRICS = [
  { label: 'Total Latency', value: '2.5s', description: 'End-to-end from task to compute' },
  { label: 'Proof Generation', value: '2.3s', description: 'HyperKZG proof creation' },
  { label: 'On-chain Finality', value: '<1s', description: 'Deterministic confirmation' },
  { label: 'Transaction Cost', value: '$0.02', description: 'USDC gas fees' },
  { label: 'Proof Size', value: '48KB', description: 'Compact SNARK proof' },
  { label: 'Purchase Amount', value: '$88', description: '4 hours A100 GPU' },
];

export default function ComputeAgentCaseStudy() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="border-b border-gray-800 bg-[#0d1117]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <Link
            href="/case-studies"
            className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Case Studies
          </Link>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center">
              <Cpu className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Agent X Purchases Compute</h1>
              <p className="text-gray-400">
                Autonomous GPU procurement with cryptographic spending proofs
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Scenario Overview */}
        <section className="mb-12">
          <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Scenario</h2>
            <p className="text-gray-400 mb-4">
              Agent X is an autonomous ML training agent. When given a new model training task,
              it needs to procure GPU compute from a decentralized marketplace. The agent operates
              with a spending policy that limits purchases to verified compute providers and enforces
              budget constraints.
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Agent Type</div>
                <div className="font-medium">ML Training Agent</div>
              </div>
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Purchase Category</div>
                <div className="font-medium">GPU Compute</div>
              </div>
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Daily Budget</div>
                <div className="font-medium">$500 USDC</div>
              </div>
            </div>
          </div>
        </section>

        {/* Metrics Grid */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-6">Performance Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {METRICS.map((metric) => (
              <div key={metric.label} className="bg-[#0d1117] border border-gray-800 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-purple-400 mb-1">{metric.value}</div>
                <div className="text-sm font-medium text-white">{metric.label}</div>
                <div className="text-xs text-gray-500 mt-1">{metric.description}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Architecture Diagram */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-6">Transaction Flow</h2>
          <ArchitectureDiagram variant="full" showTimings={true} />
        </section>

        {/* Step-by-Step Timeline */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-6">Step-by-Step Execution</h2>
          <div className="space-y-4">
            {TIMELINE_STEPS.map((step, index) => (
              <div key={step.step} className="bg-[#0d1117] border border-gray-800 rounded-xl overflow-hidden">
                <div className="flex items-center gap-4 p-4 border-b border-gray-800">
                  <div className="w-10 h-10 bg-purple-500/10 rounded-full flex items-center justify-center text-purple-400">
                    {step.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 font-mono">{step.timing}</span>
                      <h3 className="font-semibold">{step.title}</h3>
                    </div>
                    <p className="text-sm text-gray-400">{step.description}</p>
                  </div>
                  {index < TIMELINE_STEPS.length - 1 && (
                    <ArrowRight className="w-5 h-5 text-gray-600" />
                  )}
                </div>
                <div className="p-4 bg-gray-900/50">
                  <pre className="text-xs text-gray-400 overflow-x-auto">
                    <code>{step.code}</code>
                  </pre>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Why Arc is Required */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Why This Only Works on Arc
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Aspect</th>
                  <th className="text-left py-3 px-4 text-green-400 font-medium">Arc</th>
                  <th className="text-left py-3 px-4 text-red-400 font-medium">Ethereum/L2s</th>
                  <th className="text-left py-3 px-4 text-amber-400 font-medium">Why It Matters</th>
                </tr>
              </thead>
              <tbody>
                {COMPARISON_TABLE.map((row, index) => (
                  <tr
                    key={row.aspect}
                    className={`border-b border-gray-800/50 ${index % 2 === 0 ? 'bg-[#0d1117]' : ''}`}
                  >
                    <td className="py-3 px-4 font-medium text-white">{row.aspect}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2 text-green-400">
                        <CheckCircle className="w-4 h-4" />
                        {row.arc}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2 text-red-400">
                        <XCircle className="w-4 h-4" />
                        {row.ethereum}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-400">{row.issue}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Production Considerations */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-6">Production Deployment</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Server className="w-5 h-5 text-purple-400" />
                Infrastructure Requirements
              </h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Jolt prover instance (8GB+ RAM)
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Arc testnet RPC access
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Agent wallet with USDC balance
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Policy model trained on spending data
                </li>
              </ul>
            </div>
            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-cyan-400" />
                Security Checklist
              </h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Register model hash in PolicyRegistry
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Set appropriate expiry windows
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Implement nonce tracking
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Monitor attestation events
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Code Example */}
        <section>
          <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
            <Code className="w-5 h-5 text-purple-400" />
            Full Integration Example
          </h2>
          <div className="bg-[#0d1117] border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-4 py-2 border-b border-gray-800 bg-gray-900/50">
              <span className="text-xs text-gray-400">compute-purchase.ts</span>
            </div>
            <pre className="p-4 text-sm text-gray-400 overflow-x-auto">
              <code>{`import { SpendingProofsSDK } from '@arc/spending-proofs';
import { createWalletClient, http } from 'viem';
import { arcTestnet } from '@arc/chains';

// Initialize SDK
const sdk = new SpendingProofsSDK({
  proverUrl: process.env.PROVER_URL,
  chain: arcTestnet,
});

// Agent's spending policy
const policy = {
  maxSinglePurchase: 500,
  allowedCategories: ['compute', 'storage', 'api'],
  minMerchantTrust: 0.7,
};

async function purchaseCompute(task: Task) {
  // 1. Find best compute offer
  const offer = await marketplace.getBestOffer({
    requirements: task.computeRequirements,
    maxPrice: policy.maxSinglePurchase,
  });

  // 2. Prepare transaction inputs
  const inputs = {
    price: offer.totalPrice,
    merchantTrust: offer.provider.trustScore,
    categoryMatch: policy.allowedCategories.includes('compute') ? 1 : 0,
    urgency: task.deadline ? calculateUrgency(task.deadline) : 0.5,
    volatility: offer.priceVolatility,
    stockLevel: offer.availability,
    promotionalDiscount: offer.discount || 0,
    historicalAccuracy: await getHistoricalAccuracy(offer.provider),
  };

  // 3. Generate spending proof
  const { proof, decision } = await sdk.generateProof(inputs);

  if (decision !== 'approve') {
    throw new Error('Spending policy rejected this purchase');
  }

  // 4. Create transaction intent
  const txIntent = sdk.createTxIntent({
    sender: wallet.address,
    recipient: offer.provider.address,
    amountUsdc: offer.totalPrice,
    policyId: 'compute-agent-v1',
  });

  // 5. Execute gated transfer on Arc
  const tx = await sdk.gatedTransfer({
    recipient: offer.provider.address,
    amountUsdc: offer.totalPrice,
    proofHash: proof.hash,
    txIntent,
  });

  // 6. Wait for finality (<1 second on Arc)
  const receipt = await tx.wait();

  return {
    transactionHash: receipt.hash,
    explorerUrl: \`https://testnet.arcscan.app/tx/\${receipt.hash}\`,
    computeInstance: await offer.provider.provision(receipt.hash),
  };
}`}</code>
            </pre>
          </div>
        </section>
      </div>
    </div>
  );
}
