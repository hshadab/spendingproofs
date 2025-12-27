'use client';

import Link from 'next/link';
import { ArrowLeft, Shield, Lock } from 'lucide-react';
import { EnforcementDemo } from '@/components/EnforcementDemo';
import { PerformanceMetrics } from '@/components/PerformanceMetrics';
import { PolicyRegistryPanel } from '@/components/PolicyRegistryPanel';

export default function EnforcementPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/demo" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-4 h-4" />
              Back to Demos
            </Link>
          </div>
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5" />
            </div>
            <span className="font-semibold">Spending Proofs</span>
          </Link>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-full px-3 py-1 text-sm text-red-400 mb-4">
            <Lock className="w-4 h-4" />
            Hard Enforcement
          </div>
          <h1 className="text-3xl font-bold mb-3">SpendingGate Enforcement</h1>
          <p className="text-gray-400">
            This demo shows how SpendingGate <strong className="text-white">enforces</strong> proof requirements.
            Unlike simple attestation (logging), enforcement means transactions <strong className="text-red-400">REVERT</strong> without valid proofs.
          </p>
        </div>

        {/* Attestation vs Enforcement */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
            <h3 className="font-semibold mb-2 text-gray-300">Attestation (Current)</h3>
            <p className="text-sm text-gray-400 mb-3">
              Proof is logged on-chain for auditability. Transfer proceeds regardless of proof status.
            </p>
            <div className="text-xs text-amber-400 bg-amber-500/10 px-2 py-1 rounded inline-block">
              Advisory only
            </div>
          </div>
          <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-5">
            <h3 className="font-semibold mb-2 text-purple-300">Enforcement (This Demo)</h3>
            <p className="text-sm text-gray-400 mb-3">
              SpendingGate contract checks proof validity. Transfer reverts if proof is missing or invalid.
            </p>
            <div className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded inline-block">
              Mandatory gating
            </div>
          </div>
        </div>

        {/* Main Demo */}
        <EnforcementDemo />

        {/* Technical Note */}
        <div className="mt-8 bg-[#0d1117] border border-gray-800 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Implementation Note</h3>
          <p className="text-sm text-gray-400 mb-4">
            This demo simulates the SpendingGate contract interface. In production:
          </p>
          <ul className="space-y-2 text-sm text-gray-400">
            <li className="flex items-start gap-2">
              <span className="text-purple-400">1.</span>
              Full on-chain SNARK verification using HyperKZG verifier contract
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">2.</span>
              BN254 pairing precompile for efficient verification (~300k gas)
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">3.</span>
              PolicyRegistry contract for approved model/VK hash management
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">4.</span>
              Integration with Arc&apos;s USDC transfer hooks
            </li>
          </ul>
        </div>

        {/* Policy Registry */}
        <div className="mt-8">
          <PolicyRegistryPanel variant="detailed" />
        </div>

        {/* Performance */}
        <div className="mt-8">
          <PerformanceMetrics variant="compact" />
        </div>
      </div>
    </div>
  );
}
