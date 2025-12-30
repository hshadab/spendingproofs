'use client';

import { useState } from 'react';
import { XCircle, DollarSign, Zap, Lock, Globe, ChevronDown, ChevronUp, BarChart3 } from 'lucide-react';
import { ComparisonTable } from './ComparisonTable';

export function WhyArcSection() {
  const [showComparison, setShowComparison] = useState(false);

  return (
    <section id="why-arc" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Why Agents Require Arc</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            These aren&apos;t nice-to-haves. Autonomous agents break on other chains.
            Arc&apos;s architecture is the only viable foundation for agent commerce.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-[#0a0a0a] border border-red-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <XCircle className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-red-300">Without Predictable Fees</h3>
                <p className="text-sm text-gray-400">
                  Agents budget in USDC. A 10x gas spike on Ethereum bankrupts the agent mid-transaction.
                  The agent can&apos;t complete its task, user funds are stuck, trust is broken.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-green-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <DollarSign className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-green-300">Arc: USDC-Native Gas</h3>
                <p className="text-sm text-gray-400">
                  Fees paid in USDC. Agents budget accurately. No token swaps, no volatility exposure,
                  no failed transactions from gas estimation errors.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-red-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <XCircle className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-red-300">Without Deterministic Finality</h3>
                <p className="text-sm text-gray-400">
                  A shopping agent waits 12 confirmations on Ethereum. The deal expires.
                  On Solana, a reorg reverses the payment. The agent already shipped the goods.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-green-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <Zap className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-green-300">Arc: Sub-Second Deterministic Finality</h3>
                <p className="text-sm text-gray-400">
                  Transaction is final. No reorgs, no probabilistic waiting. Agents chain operations
                  instantly—proof, payment, delivery in one atomic flow.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-red-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <XCircle className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-red-300">Without Opt-in Privacy</h3>
                <p className="text-sm text-gray-400">
                  Agent spending patterns are public. Competitors front-run deals.
                  MEV bots extract value from every agent transaction. Strategies are exposed.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-green-500/20 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <Lock className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-green-300">Arc: Opt-in Privacy</h3>
                <p className="text-sm text-gray-400">
                  Confidential transactions when strategy matters. Public when transparency is needed.
                  Agents choose per-transaction. No MEV extraction, no front-running.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Comparison Table Toggle */}
        <div className="mb-8">
          <button
            onClick={() => setShowComparison(!showComparison)}
            className="w-full flex items-center justify-center gap-2 py-4 px-6 bg-[#0a0a0a] border border-gray-800 hover:border-purple-500/30 rounded-xl transition-colors group"
          >
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span className="font-medium text-gray-300 group-hover:text-white">
              {showComparison ? 'Hide' : 'View'} Full Comparison: Arc vs Other L2s
            </span>
            {showComparison ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </button>

          {showComparison && (
            <div className="mt-4 p-6 bg-[#0a0a0a] border border-gray-800 rounded-xl">
              <ComparisonTable variant="full" showScenarios={true} />
            </div>
          )}
        </div>

        {/* Enterprise Rails callout */}
        <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-500/20 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
              <Globe className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="font-semibold mb-2">Enterprise-Grade Rails Required</h3>
              <p className="text-sm text-gray-400">
                Enterprise agents need compliance, audit trails, and institutional custody.
                Arc&apos;s Circle-backed infrastructure with StableFX and Payments Network provides
                production-grade rails that enterprises actually deploy on—not testnet experiments.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
