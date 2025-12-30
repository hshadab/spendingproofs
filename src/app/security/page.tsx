'use client';

import Link from 'next/link';
import {
  Shield,
  Lock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowLeft,
  FileCheck,
  Key,
  RefreshCcw,
  Eye,
  Fingerprint,
  Server,
  Zap,
} from 'lucide-react';

interface ThreatVector {
  id: string;
  name: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  mitigation: string;
  protection: string;
}

const THREAT_VECTORS: ThreatVector[] = [
  {
    id: 'input-tampering',
    name: 'Input Tampering',
    description: 'Attacker modifies transaction inputs after proof generation to trick merchant into accepting fraudulent transaction.',
    severity: 'critical',
    mitigation: 'inputsHash',
    protection: 'All 8 model inputs are hashed and included in the proof. Any change to inputs invalidates the proof.',
  },
  {
    id: 'proof-replay',
    name: 'Proof Replay',
    description: 'Attacker reuses a valid proof for a different transaction or after the original was canceled.',
    severity: 'critical',
    mitigation: 'txIntentHash + nonce',
    protection: 'Each proof is bound to specific txIntent (recipient, amount, expiry). Nonces prevent reuse.',
  },
  {
    id: 'model-substitution',
    name: 'Model Substitution',
    description: 'Agent uses a permissive model that approves everything, bypassing intended policy constraints.',
    severity: 'high',
    mitigation: 'PolicyRegistry',
    protection: 'On-chain PolicyRegistry maps agents to approved model hashes. SpendingGate rejects unregistered models.',
  },
  {
    id: 'front-running',
    name: 'Front-Running / MEV',
    description: 'Adversary observes pending transaction and front-runs to capture value or information.',
    severity: 'high',
    mitigation: 'Opt-in privacy',
    protection: 'Arc supports confidential transactions. Agent can choose private submission for sensitive trades.',
  },
  {
    id: 'proof-forgery',
    name: 'Proof Forgery',
    description: 'Attacker creates fake proof without running the actual model.',
    severity: 'critical',
    mitigation: 'HyperKZG verification',
    protection: 'SNARK proofs are cryptographically unforgeable. Verifier accepts only valid computations.',
  },
  {
    id: 'expired-proof',
    name: 'Expired Proof Usage',
    description: 'Attacker uses old proof after conditions have changed (price, inventory, etc).',
    severity: 'medium',
    mitigation: 'Expiry timestamp',
    protection: 'Each txIntent includes expiry. SpendingGate rejects proofs past their expiration.',
  },
  {
    id: 'agent-impersonation',
    name: 'Agent Impersonation',
    description: 'Attacker claims to be a different agent to use their policy limits.',
    severity: 'high',
    mitigation: 'Wallet signature',
    protection: 'Transaction must be signed by agent wallet. Policy applies to msg.sender only.',
  },
  {
    id: 'prover-manipulation',
    name: 'Prover Manipulation',
    description: 'Malicious prover returns invalid proofs or leaks input data.',
    severity: 'medium',
    mitigation: 'Proof verification',
    protection: 'On-chain verification validates proof correctness. Self-hosting available for sensitive data.',
  },
];

interface TrustAssumption {
  component: string;
  assumption: string;
  risk: string;
  mitigation: string;
}

const TRUST_ASSUMPTIONS: TrustAssumption[] = [
  {
    component: 'Jolt Prover',
    assumption: 'Prover generates valid proofs',
    risk: 'Malicious prover could attempt to generate invalid proofs',
    mitigation: 'On-chain verification; self-host option available',
  },
  {
    component: 'Arc Network',
    assumption: 'Chain provides correct state and finality',
    risk: 'Consensus failure could affect transaction ordering',
    mitigation: 'Circle-backed infrastructure; deterministic finality',
  },
  {
    component: 'PolicyRegistry',
    assumption: 'Registry correctly maps agents to approved policies',
    risk: 'Misconfiguration could allow unauthorized models',
    mitigation: 'Multi-sig governance; timelock on updates',
  },
  {
    component: 'USDC Contract',
    assumption: 'USDC transfers execute correctly',
    risk: 'Contract bug could affect transfers',
    mitigation: 'Circle-audited native USDC; battle-tested',
  },
  {
    component: 'Model Weights',
    assumption: 'Model was trained correctly and is deterministic',
    risk: 'Biased or incorrect model could make wrong decisions',
    mitigation: 'Model hash commitment; version control; audits',
  },
];

const SECURITY_GUARANTEES = [
  {
    category: 'Cryptographic',
    items: [
      'HyperKZG proofs are computationally infeasible to forge',
      'Input hash binding prevents post-generation tampering',
      'Transaction intent hash prevents cross-context replay',
    ],
  },
  {
    category: 'Economic',
    items: [
      'USDC stability eliminates gas volatility attack surface',
      'No MEV extraction via opt-in privacy',
      'Failed transactions still cost gas, discouraging spam',
    ],
  },
  {
    category: 'Operational',
    items: [
      'Sub-second finality prevents confirmation gaming',
      'Expiry timestamps limit proof validity window',
      'Nonce tracking prevents transaction replay',
    ],
  },
];

function SeverityBadge({ severity }: { severity: ThreatVector['severity'] }) {
  const colors = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    low: 'bg-green-500/20 text-green-400 border-green-500/30',
  };

  return (
    <span className={`text-xs px-2 py-0.5 rounded border ${colors[severity]}`}>
      {severity.toUpperCase()}
    </span>
  );
}

function ThreatCard({ threat }: { threat: ThreatVector }) {
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5 hover:border-purple-500/30 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
          <h3 className="font-semibold text-white">{threat.name}</h3>
        </div>
        <SeverityBadge severity={threat.severity} />
      </div>
      <p className="text-sm text-gray-400 mb-4">{threat.description}</p>
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Shield className="w-4 h-4 text-purple-400" />
          <span className="text-purple-400 font-medium">Mitigation:</span>
          <code className="text-xs bg-purple-500/10 px-2 py-0.5 rounded text-purple-300">
            {threat.mitigation}
          </code>
        </div>
        <p className="text-sm text-gray-500 pl-6">{threat.protection}</p>
      </div>
    </div>
  );
}

export default function SecurityPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="border-b border-gray-800 bg-[#0d1117]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center">
              <Shield className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Security Model</h1>
              <p className="text-gray-400">
                Comprehensive threat model and security guarantees for zkML spending proofs
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Overview */}
        <section className="mb-12">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-[#0d1117] border border-green-500/20 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <Fingerprint className="w-6 h-6 text-green-400" />
                <h3 className="font-semibold">Zero-Knowledge</h3>
              </div>
              <p className="text-sm text-gray-400">
                Proofs reveal only the decision, never the underlying inputs or policy thresholds.
              </p>
            </div>
            <div className="bg-[#0d1117] border border-purple-500/20 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <Lock className="w-6 h-6 text-purple-400" />
                <h3 className="font-semibold">Tamper-Proof</h3>
              </div>
              <p className="text-sm text-gray-400">
                Cryptographic commitments bind proofs to specific inputs and transaction intents.
              </p>
            </div>
            <div className="bg-[#0d1117] border border-cyan-500/20 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <Zap className="w-6 h-6 text-cyan-400" />
                <h3 className="font-semibold">Instant Finality</h3>
              </div>
              <p className="text-sm text-gray-400">
                Arc&apos;s deterministic finality eliminates confirmation attacks and reorg risks.
              </p>
            </div>
          </div>
        </section>

        {/* Threat Vectors */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-amber-400" />
            Threat Vectors & Mitigations
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {THREAT_VECTORS.map((threat) => (
              <ThreatCard key={threat.id} threat={threat} />
            ))}
          </div>
        </section>

        {/* Trust Assumptions */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Server className="w-6 h-6 text-purple-400" />
            Trust Assumptions
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Component</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Assumption</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Risk</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Mitigation</th>
                </tr>
              </thead>
              <tbody>
                {TRUST_ASSUMPTIONS.map((item, index) => (
                  <tr
                    key={item.component}
                    className={`border-b border-gray-800/50 ${index % 2 === 0 ? 'bg-[#0d1117]' : ''}`}
                  >
                    <td className="py-3 px-4 font-medium text-white">{item.component}</td>
                    <td className="py-3 px-4 text-gray-400">{item.assumption}</td>
                    <td className="py-3 px-4 text-amber-400">{item.risk}</td>
                    <td className="py-3 px-4 text-green-400">{item.mitigation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Security Guarantees */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-400" />
            Security Guarantees
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {SECURITY_GUARANTEES.map((category) => (
              <div key={category.category} className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">{category.category}</h3>
                <ul className="space-y-3">
                  {category.items.map((item, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-gray-400">
                      <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Hash Commitments Diagram */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Key className="w-6 h-6 text-cyan-400" />
            Proof Binding Architecture
          </h2>
          <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
            <div className="grid md:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-xs text-gray-500 mb-2">inputsHash</div>
                <code className="text-sm text-purple-400 break-all">
                  keccak256(price, trust, category, urgency, volatility, stock, discount, accuracy)
                </code>
                <p className="text-xs text-gray-500 mt-2">Prevents input tampering</p>
              </div>
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-xs text-gray-500 mb-2">modelHash</div>
                <code className="text-sm text-cyan-400 break-all">
                  keccak256(model_weights)
                </code>
                <p className="text-xs text-gray-500 mt-2">Locks to specific model</p>
              </div>
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-xs text-gray-500 mb-2">txIntentHash</div>
                <code className="text-sm text-amber-400 break-all">
                  keccak256(chainId, usdc, sender, recipient, amount, nonce, expiry, policyId)
                </code>
                <p className="text-xs text-gray-500 mt-2">Binds to transaction</p>
              </div>
              <div className="p-4 bg-gray-900 rounded-lg">
                <div className="text-xs text-gray-500 mb-2">proofHash</div>
                <code className="text-sm text-green-400 break-all">
                  keccak256(proof_bytes)
                </code>
                <p className="text-xs text-gray-500 mt-2">On-chain attestation</p>
              </div>
            </div>
          </div>
        </section>

        {/* Attack Scenarios */}
        <section>
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <XCircle className="w-6 h-6 text-red-400" />
            Attack Scenario Analysis
          </h2>
          <div className="space-y-4">
            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">1.</span>
                <h3 className="font-semibold">Price Change After Proof</h3>
              </div>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-500 mb-1">Attack:</div>
                  <p className="text-gray-400">Agent generates proof for $100 item. Before submission, price changes to $150.</p>
                </div>
                <div>
                  <div className="text-green-500 mb-1">Defense:</div>
                  <p className="text-gray-400">inputsHash includes price. Proof with $100 price won&apos;t match $150 on-chain state. Merchant verification fails.</p>
                </div>
              </div>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">2.</span>
                <h3 className="font-semibold">Proof Stolen and Reused</h3>
              </div>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-500 mb-1">Attack:</div>
                  <p className="text-gray-400">Attacker intercepts valid proof and attempts to use it for their own transaction.</p>
                </div>
                <div>
                  <div className="text-green-500 mb-1">Defense:</div>
                  <p className="text-gray-400">txIntentHash includes sender address. Proof only valid when msg.sender matches original agent.</p>
                </div>
              </div>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">3.</span>
                <h3 className="font-semibold">Permissive Model Substitution</h3>
              </div>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-500 mb-1">Attack:</div>
                  <p className="text-gray-400">Malicious agent uses model that approves everything regardless of inputs.</p>
                </div>
                <div>
                  <div className="text-green-500 mb-1">Defense:</div>
                  <p className="text-gray-400">PolicyRegistry on-chain maps agent to approved modelHash. SpendingGate rejects proofs from unregistered models.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
