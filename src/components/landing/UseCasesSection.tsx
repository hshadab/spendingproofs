'use client';

import { ShoppingCart, Gauge, GitBranch, Globe } from 'lucide-react';

export function UseCasesSection() {
  return (
    <section id="use-cases" className="py-16 px-6 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Real Use Cases Today</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Not hypothetical. These agents exist. They need policy proofs to unlock real economic autonomy.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <ShoppingCart className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2">AI Shopping Agents</h3>
                <p className="text-sm text-gray-400 mb-3">
                  Autonomous agents that browse, compare, and purchase products on behalf of users.
                  Need cryptographic proof they stayed within budget before merchant releases goods.
                </p>
                <div className="text-xs text-purple-400">
                  Arc requirement: Sub-second finality for instant checkout
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-cyan-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <Gauge className="w-6 h-6 text-cyan-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2">Subscription Managers</h3>
                <p className="text-sm text-gray-400 mb-3">
                  Agents that optimize SaaS spend by upgrading, downgrading, or canceling subscriptions.
                  Proof ensures agent followed cost-saving rules, not upsell prompts.
                </p>
                <div className="text-xs text-cyan-400">
                  Arc requirement: USDC gas for predictable monthly costs
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-amber-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <GitBranch className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2">Multi-Agent Marketplaces</h3>
                <p className="text-sm text-gray-400 mb-3">
                  Agents hiring other agents for subtasks. Each agent proves it followed its delegated
                  budget. Parent agent verifies before releasing child agent payments.
                </p>
                <div className="text-xs text-amber-400">
                  Arc requirement: Deterministic finality for chained payments
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <Globe className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <h3 className="font-semibold mb-2">Enterprise Expense Agents</h3>
                <p className="text-sm text-gray-400 mb-3">
                  Corporate agents with department-level spending authority. Proof provides audit trail
                  that agent followed company policy—not just approval, but cryptographic compliance.
                </p>
                <div className="text-xs text-green-400">
                  Arc requirement: Enterprise rails with Circle integration
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
