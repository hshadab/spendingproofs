'use client';

import { Zap, Shield, Lock, Layers } from 'lucide-react';

export function ArcValueProps() {
  return (
    <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/10 rounded-xl border border-purple-200 dark:border-purple-800 p-6">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
        Why Arc for AI Agents?
      </h3>

      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-purple-100 dark:border-purple-900">
          <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-3">
            <Zap className="w-5 h-5 text-purple-600" />
          </div>
          <h4 className="font-semibold text-slate-900 dark:text-white mb-1">
            Stablecoin Gas
          </h4>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Pay gas in USDC. No volatile ETH needed for agent operations.
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-purple-100 dark:border-purple-900">
          <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-3">
            <Shield className="w-5 h-5 text-purple-600" />
          </div>
          <h4 className="font-semibold text-slate-900 dark:text-white mb-1">
            Sub-Second Finality
          </h4>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Payments confirm in under 1 second. Real-time agent decisions.
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-purple-100 dark:border-purple-900">
          <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-3">
            <Lock className="w-5 h-5 text-purple-600" />
          </div>
          <h4 className="font-semibold text-slate-900 dark:text-white mb-1">
            Privacy Preserving
          </h4>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            zkML proves execution without revealing proprietary inputs.
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border border-purple-100 dark:border-purple-900">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center flex-shrink-0">
            <Layers className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-1">
              Composable Primitive
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Other Arc projects can use this proof attestation system: DeFi protocols requiring
              verified AI decisions, insurance claims with auditable ML assessments, trading bots
              with provable strategy execution.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
