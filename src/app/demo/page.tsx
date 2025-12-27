'use client';

import Link from 'next/link';
import { Play, CreditCard, AlertTriangle, ArrowLeft, Shield } from 'lucide-react';

export default function DemoHub() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-4 h-4" />
              Back to SDK
            </Link>
          </div>
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5" />
            </div>
            <span className="font-semibold">Arc Policy Proofs</span>
          </Link>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Interactive Demos</h1>
          <p className="text-gray-400 text-lg">
            Experience zkML spending proofs in action. Try the playground,
            run an end-to-end payment, or test tamper detection.
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
                  <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">Recommended</span>
                </div>
                <p className="text-gray-400 mb-4">
                  Configure spending policies with sliders, simulate purchase scenarios,
                  and watch real SNARK proofs generate in 4-12 seconds.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Policy Config</span>
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Real Proofs</span>
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Decision Visualization</span>
                </div>
              </div>
            </div>
          </Link>

          {/* Payment */}
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
                  <h2 className="text-xl font-semibold">End-to-End Payment</h2>
                  <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">Requires Wallet</span>
                </div>
                <p className="text-gray-400 mb-4">
                  Connect your wallet and run the full flow: policy check, proof generation,
                  on-chain attestation, and USDC payment on Arc Testnet.
                </p>
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">Wallet Connect</span>
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">On-Chain Attestation</span>
                  <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">USDC Transfer</span>
                </div>
              </div>
            </div>
          </Link>

          {/* Tamper */}
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
                  <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">Security Demo</span>
                </div>
                <p className="text-gray-400 mb-4">
                  See how modifying inputs after proof generation causes verification to fail.
                  Side-by-side comparison of original vs tampered inputs.
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
          <div className="grid md:grid-cols-2 gap-6 text-sm">
            <div>
              <h4 className="text-gray-400 mb-2">Proof System</h4>
              <ul className="space-y-1 text-gray-500">
                <li>JOLT-Atlas SNARK (HyperKZG/BN254)</li>
                <li>45-55KB proof size</li>
                <li>4-12 second generation</li>
                <li>&lt;150ms verification</li>
              </ul>
            </div>
            <div>
              <h4 className="text-gray-400 mb-2">Arc Testnet</h4>
              <ul className="space-y-1 text-gray-500">
                <li>Chain ID: 5042002</li>
                <li>USDC native gas</li>
                <li>Sub-second finality</li>
                <li>Attestation contract deployed</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
