'use client';

import { Bot, Cpu, ShieldCheck, FileCheck, Wallet, ArrowRight, CheckCircle2, Lock, Eye } from 'lucide-react';

const flowSteps = [
  {
    id: 1,
    title: 'Purchase Intent',
    description: 'Agent identifies a purchase opportunity and gathers context',
    icon: Bot,
    color: 'blue',
    details: [
      'Price: $0.05 USDC',
      'Merchant: api-service.com',
      'Category: API calls',
      'Remaining budget: $1.00',
    ],
  },
  {
    id: 2,
    title: 'Generate Proof',
    description: 'Agent calls prover with spending context, zkML model runs inside SNARK',
    icon: Cpu,
    color: 'purple',
    details: [
      'ML model evaluates policy',
      'JOLT-Atlas generates SNARK',
      'Proof includes program_io',
      'Returns proofHash + decision',
    ],
  },
  {
    id: 3,
    title: 'Verify Proof',
    description: 'Cryptographic verification gates the transfer (trust boundary)',
    icon: ShieldCheck,
    color: 'green',
    details: [
      'Verify SNARK is valid',
      'Check program_io matches',
      'Confirm decision = approved',
      'BLOCKS transfer if invalid',
    ],
    isTrustBoundary: true,
  },
  {
    id: 4,
    title: 'Attest On-Chain',
    description: 'Record proof hash on-chain for transparency and audit trail',
    icon: FileCheck,
    color: 'amber',
    details: [
      'Submit to ProofAttestation',
      'Immutable audit record',
      'Anyone can verify later',
      'Transparency layer',
    ],
  },
  {
    id: 5,
    title: 'Execute Transfer',
    description: 'SpendingGateWallet releases USDC after checking attestation',
    icon: Wallet,
    color: 'emerald',
    details: [
      'Check proof is attested',
      'Verify nonce unused',
      'Check expiry valid',
      'Transfer USDC to merchant',
    ],
  },
];

const colorClasses = {
  blue: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30' },
  purple: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
  green: { bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/30' },
  amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
  emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
};

export function AgentFlowSection() {
  return (
    <section id="agent-flow" className="py-16 px-6 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Complete Agent Transaction Flow</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Every agent purchase follows this cryptographically enforced pipeline.
            Verification is the trust boundary—attestation provides transparency.
          </p>
        </div>

        {/* Trust Boundary Legend */}
        <div className="flex justify-center gap-8 mb-10">
          <div className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-gray-400">Trust Boundary (gates transfer)</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded-full bg-amber-500"></div>
            <span className="text-gray-400">Transparency Layer (audit trail)</span>
          </div>
        </div>

        {/* Flow Steps - Horizontal on large screens */}
        <div className="hidden lg:block">
          <div className="flex items-start justify-between relative">
            {/* Connection line */}
            <div className="absolute top-12 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500 opacity-30"></div>

            {flowSteps.map((step, index) => {
              const colors = colorClasses[step.color as keyof typeof colorClasses];
              const Icon = step.icon;

              return (
                <div key={step.id} className="flex flex-col items-center relative" style={{ width: '18%' }}>
                  {/* Step number and icon */}
                  <div className={`relative z-10 w-24 h-24 rounded-2xl ${colors.bg} ${step.isTrustBoundary ? 'ring-2 ring-green-500' : ''} flex flex-col items-center justify-center mb-4`}>
                    <span className={`text-xs font-mono ${colors.text} mb-1`}>Step {step.id}</span>
                    <Icon className={`w-8 h-8 ${colors.text}`} />
                  </div>

                  {/* Arrow (except last) */}
                  {index < flowSteps.length - 1 && (
                    <div className="absolute top-10 -right-4 z-20">
                      <ArrowRight className="w-6 h-6 text-gray-600" />
                    </div>
                  )}

                  {/* Trust boundary indicator */}
                  {step.isTrustBoundary && (
                    <div className="absolute -top-2 -right-2 z-20">
                      <div className="bg-green-500 rounded-full p-1">
                        <Lock className="w-3 h-3 text-black" />
                      </div>
                    </div>
                  )}

                  {/* Content */}
                  <h3 className={`font-semibold text-center mb-2 ${colors.text}`}>{step.title}</h3>
                  <p className="text-xs text-gray-500 text-center mb-3 px-2">{step.description}</p>

                  {/* Details */}
                  <div className={`bg-[#0d1117] border ${colors.border} rounded-lg p-3 w-full`}>
                    <ul className="space-y-1">
                      {step.details.map((detail, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs">
                          <CheckCircle2 className={`w-3 h-3 ${colors.text} mt-0.5 flex-shrink-0`} />
                          <span className="text-gray-400">{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Flow Steps - Vertical on mobile */}
        <div className="lg:hidden space-y-6">
          {flowSteps.map((step, index) => {
            const colors = colorClasses[step.color as keyof typeof colorClasses];
            const Icon = step.icon;

            return (
              <div key={step.id}>
                <div className={`bg-[#0d1117] border ${colors.border} ${step.isTrustBoundary ? 'ring-2 ring-green-500' : ''} rounded-xl p-5`}>
                  <div className="flex items-start gap-4">
                    <div className={`w-14 h-14 rounded-xl ${colors.bg} flex items-center justify-center flex-shrink-0`}>
                      <Icon className={`w-7 h-7 ${colors.text}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs font-mono ${colors.text}`}>Step {step.id}</span>
                        {step.isTrustBoundary && (
                          <span className="bg-green-500/20 text-green-400 text-xs px-2 py-0.5 rounded-full flex items-center gap-1">
                            <Lock className="w-3 h-3" /> Trust Boundary
                          </span>
                        )}
                      </div>
                      <h3 className="font-semibold mb-1">{step.title}</h3>
                      <p className="text-sm text-gray-400 mb-3">{step.description}</p>
                      <ul className="space-y-1">
                        {step.details.map((detail, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm">
                            <CheckCircle2 className={`w-4 h-4 ${colors.text} mt-0.5 flex-shrink-0`} />
                            <span className="text-gray-400">{detail}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
                {index < flowSteps.length - 1 && (
                  <div className="flex justify-center py-2">
                    <ArrowRight className="w-5 h-5 text-gray-600 rotate-90" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Key Insight Box */}
        <div className="mt-12 bg-gradient-to-r from-green-500/5 to-amber-500/5 border border-gray-800 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500/20 to-amber-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <div>
              <h4 className="font-semibold mb-2">Understanding the Architecture</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                <strong className="text-green-400">Verification</strong> is where security happens—it cryptographically
                validates the proof and <em>blocks</em> the transfer if invalid.
                <strong className="text-amber-400 ml-1">Attestation</strong> is about transparency—recording an immutable
                audit trail on-chain so anyone can verify the agent followed its policy.
                The SpendingGateWallet only releases funds after checking both.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
