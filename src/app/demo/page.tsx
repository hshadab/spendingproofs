'use client';

import Link from 'next/link';
import { Play, CreditCard, AlertTriangle, Lock, Eye } from 'lucide-react';

export default function DemoHub() {
  return (
    <div className="max-w-4xl mx-auto py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Interactive Demos</h1>
        <p className="text-gray-400 text-lg">
          Experience zkML spending proofs in action. Generate real proofs,
          see the privacy model, and explore enforcement scenarios.
        </p>
      </div>

      <div className="grid gap-6">
        {/* Playground */}
        <Link
          href="/demo/playground"
          className="group p-6 bg-[#0d1117] border border-gray-800 rounded-xl hover:border-purple-500/50 transition-colors"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center group-hover:bg-purple-500/20 transition-colors">
              <Play className="w-6 h-6 text-purple-400" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-semibold">Interactive Playground</h2>
                <span className="text-xs text-purple-400 bg-purple-500/10 px-2 py-1 rounded">Start Here</span>
              </div>
              <p className="text-gray-400 mb-4">
                Configure spending policies, simulate purchases, and generate cryptographic proofs.
                Toggle between Agent and Verifier views to see what stays private vs public.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Policy Config</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Real Proofs</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded flex items-center gap-1">
                  <Eye className="w-3 h-3" />
                  Split View
                </span>
              </div>
            </div>
          </div>
        </Link>

        {/* Payment + Enforcement */}
        <Link
          href="/demo/payment"
          className="group p-6 bg-[#0d1117] border border-gray-800 rounded-xl hover:border-green-500/50 transition-colors"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center group-hover:bg-green-500/20 transition-colors">
              <CreditCard className="w-6 h-6 text-green-400" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-semibold">Payment Flow</h2>
                <span className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded flex items-center gap-1">
                  <Lock className="w-3 h-3" />
                  No Proof, No Payment
                </span>
              </div>
              <p className="text-gray-400 mb-4">
                End-to-end flow: connect wallet, generate proof, submit attestation to Arc, execute USDC payment.
                Explore SpendingGate enforcement—transactions REVERT without valid proofs.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Wallet Connect</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">On-Chain Attestation</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Enforcement Demo</span>
              </div>
            </div>
          </div>
        </Link>

        {/* Tamper Detection */}
        <Link
          href="/demo/tamper"
          className="group p-6 bg-[#0d1117] border border-gray-800 rounded-xl hover:border-amber-500/50 transition-colors"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center group-hover:bg-amber-500/20 transition-colors">
              <AlertTriangle className="w-6 h-6 text-amber-400" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-semibold">Tamper Detection</h2>
                <span className="text-xs text-amber-400 bg-amber-500/10 px-2 py-1 rounded">Security</span>
              </div>
              <p className="text-gray-400 mb-4">
                See why proofs matter: modify inputs after proof generation and watch verification fail.
                The inputsHash binding prevents any tampering with transaction data.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Hash Comparison</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Tamper Simulation</span>
                <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Verification Failure</span>
              </div>
            </div>
          </div>
        </Link>
      </div>

      {/* Technical Info */}
      <div className="mt-12 p-6 bg-[#0d1117] border border-gray-800 rounded-xl">
        <h3 className="font-semibold mb-4">Technical Details</h3>
        <div className="grid md:grid-cols-3 gap-6 text-sm">
          <div>
            <h4 className="text-gray-400 mb-2">Proof System</h4>
            <ul className="space-y-1 text-gray-500">
              <li>JOLT-Atlas SNARK (HyperKZG/BN254)</li>
              <li>~48KB proof size</li>
              <li>2.1s p50 / 3.8s p90 proving</li>
              <li>45ms verification</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-400 mb-2">Arc Testnet</h4>
            <ul className="space-y-1 text-gray-500">
              <li>Chain ID: 5042002</li>
              <li>USDC native gas</li>
              <li>Sub-second finality</li>
              <li>SpendingGate enforcement</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-400 mb-2">Security Features</h4>
            <ul className="space-y-1 text-gray-500">
              <li>inputsHash tamper protection</li>
              <li>txIntentHash binding</li>
              <li>Replay protection</li>
              <li>Model substitution defense</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
