'use client';

import { type SpendingModelInput, createDefaultInput } from '@/lib/spendingModel';
import { RotateCcw } from 'lucide-react';

interface PurchaseSimulatorProps {
  input: SpendingModelInput;
  onChange: (input: SpendingModelInput) => void;
  disabled?: boolean;
}

export function PurchaseSimulator({ input, onChange, disabled }: PurchaseSimulatorProps) {
  const handleReset = () => {
    onChange(createDefaultInput());
  };

  return (
    <div className="bg-[#0d1117] rounded-xl border border-gray-800 border-gray-800 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white text-white">
          Simulate Purchase
        </h3>
        <button
          onClick={handleReset}
          disabled={disabled}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-300 dark:hover:text-gray-300 disabled:opacity-50"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Service Name */}
        <div className="col-span-2">
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Service Name
          </label>
          <input
            type="text"
            value={input.serviceName}
            onChange={(e) => onChange({ ...input, serviceName: e.target.value })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm disabled:opacity-50"
          />
        </div>

        {/* Price */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Price (USDC)
          </label>
          <input
            type="number"
            step="0.001"
            min="0"
            value={input.priceUsdc}
            onChange={(e) => onChange({ ...input, priceUsdc: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Budget */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Your Budget (USDC)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={input.budgetUsdc}
            onChange={(e) => onChange({ ...input, budgetUsdc: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Spent Today */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Spent Today (USDC)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={input.spentTodayUsdc}
            onChange={(e) => onChange({ ...input, spentTodayUsdc: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Daily Limit */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Daily Limit (USDC)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            value={input.dailyLimitUsdc}
            onChange={(e) => onChange({ ...input, dailyLimitUsdc: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Service Success Rate */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Service Success Rate
          </label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={input.serviceSuccessRate}
              onChange={(e) => onChange({ ...input, serviceSuccessRate: parseFloat(e.target.value) })}
              disabled={disabled}
              className="flex-1 h-2 bg-gray-700 bg-700 rounded-lg appearance-none cursor-pointer accent-purple-600 disabled:opacity-50"
            />
            <span className="text-sm font-mono text-white text-white w-12 text-right">
              {(input.serviceSuccessRate * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Service Total Calls */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Previous Calls
          </label>
          <input
            type="number"
            step="1"
            min="0"
            value={input.serviceTotalCalls}
            onChange={(e) => onChange({ ...input, serviceTotalCalls: parseInt(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Purchases in Category */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Recent Category Purchases
          </label>
          <input
            type="number"
            step="1"
            min="0"
            value={input.purchasesInCategory}
            onChange={(e) => onChange({ ...input, purchasesInCategory: parseInt(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>

        {/* Time Since Last Purchase */}
        <div>
          <label className="block text-sm text-gray-400 text-gray-400 mb-1">
            Time Since Last (seconds)
          </label>
          <input
            type="number"
            step="1"
            min="0"
            value={input.timeSinceLastPurchase}
            onChange={(e) => onChange({ ...input, timeSinceLastPurchase: parseInt(e.target.value) || 0 })}
            disabled={disabled}
            className="w-full px-3 py-2 bg-gray-800 bg-800 border border-gray-800 border-gray-700 rounded-lg text-white text-white text-sm font-mono disabled:opacity-50"
          />
        </div>
      </div>
    </div>
  );
}
