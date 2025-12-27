'use client';

import { type SpendingPolicy, DEFAULT_SPENDING_POLICY } from '@/lib/spendingModel';
import { RotateCcw } from 'lucide-react';

interface PolicySlidersProps {
  policy: SpendingPolicy;
  onChange: (policy: SpendingPolicy) => void;
  disabled?: boolean;
}

export function PolicySliders({ policy, onChange, disabled }: PolicySlidersProps) {
  const handleReset = () => {
    onChange(DEFAULT_SPENDING_POLICY);
  };

  return (
    <div className="bg-[#0d1117] rounded-xl border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">
          Spending Policy
        </h3>
        <button
          onClick={handleReset}
          disabled={disabled}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-300 disabled:opacity-50"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="space-y-6">
        {/* Daily Limit */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <label className="text-gray-400">Daily Limit (USDC)</label>
            <span className="font-mono text-white">
              ${policy.dailyLimitUsdc.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.50"
            max="5.00"
            step="0.10"
            value={policy.dailyLimitUsdc}
            onChange={(e) => onChange({ ...policy, dailyLimitUsdc: parseFloat(e.target.value) })}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>$0.50</span>
            <span>$5.00</span>
          </div>
        </div>

        {/* Max Single Purchase */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <label className="text-gray-400">Max Single Purchase (USDC)</label>
            <span className="font-mono text-white">
              ${policy.maxSinglePurchaseUsdc.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.50"
            step="0.01"
            value={policy.maxSinglePurchaseUsdc}
            onChange={(e) => onChange({ ...policy, maxSinglePurchaseUsdc: parseFloat(e.target.value) })}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>$0.01</span>
            <span>$0.50</span>
          </div>
        </div>

        {/* Min Success Rate */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <label className="text-gray-400">Min Service Success Rate</label>
            <span className="font-mono text-white">
              {(policy.minSuccessRate * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.30"
            max="1.00"
            step="0.05"
            value={policy.minSuccessRate}
            onChange={(e) => onChange({ ...policy, minSuccessRate: parseFloat(e.target.value) })}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>30%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Min Budget Buffer */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <label className="text-gray-400">Min Budget Buffer (USDC)</label>
            <span className="font-mono text-white">
              ${policy.minBudgetBuffer.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.00"
            max="0.10"
            step="0.01"
            value={policy.minBudgetBuffer}
            onChange={(e) => onChange({ ...policy, minBudgetBuffer: parseFloat(e.target.value) })}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>$0.00</span>
            <span>$0.10</span>
          </div>
        </div>
      </div>
    </div>
  );
}
