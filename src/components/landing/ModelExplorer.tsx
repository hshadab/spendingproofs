'use client';

import { useState } from 'react';

export function ModelExplorer() {
  const [inputs, setInputs] = useState({
    price: 0.05,
    budget: 1.0,
    spentToday: 0.2,
    dailyLimit: 0.5,
  });

  // Simple decision logic for demo (mimics the actual model behavior)
  const withinDailyLimit = inputs.spentToday + inputs.price <= inputs.dailyLimit;
  const withinBudget = inputs.price <= inputs.budget;
  const shouldBuy = withinDailyLimit && withinBudget;

  // Confidence based on how much room is left
  const budgetRatio = inputs.budget > 0 ? (inputs.budget - inputs.price) / inputs.budget : 0;
  const limitRatio = inputs.dailyLimit > 0 ? (inputs.dailyLimit - inputs.spentToday - inputs.price) / inputs.dailyLimit : 0;
  const confidence = shouldBuy ? Math.round(Math.min(budgetRatio, limitRatio) * 100) : Math.round((1 - Math.min(inputs.price / inputs.budget, 1)) * 30);

  // Risk based on how close to limits
  const riskScore = shouldBuy
    ? Math.round((1 - Math.min(budgetRatio, limitRatio)) * 50)
    : Math.round(70 + Math.random() * 20);

  return (
    <div className="relative">
      {/* Glow effect */}
      <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-purple-500 to-cyan-500 rounded-xl blur-lg opacity-40 animate-pulse" />
      <div className="relative bg-[#0d1117] border border-purple-500/30 rounded-xl p-5 w-full max-w-sm shadow-[0_0_30px_rgba(168,85,247,0.15)]">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-300">Spending Model Explorer</h4>
          <span className="text-xs text-gray-500">Interactive</span>
        </div>

        {/* Inputs */}
        <div className="space-y-3 mb-5">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Price (USDC)</span>
              <span className="font-mono text-purple-400">${inputs.price.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.50"
              step="0.01"
              value={inputs.price}
              onChange={(e) => setInputs({ ...inputs, price: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Budget (USDC)</span>
              <span className="font-mono text-purple-400">${inputs.budget.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="2.00"
              step="0.10"
              value={inputs.budget}
              onChange={(e) => setInputs({ ...inputs, budget: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Spent Today (USDC)</span>
              <span className="font-mono text-purple-400">${inputs.spentToday.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.00"
              max="1.00"
              step="0.05"
              value={inputs.spentToday}
              onChange={(e) => setInputs({ ...inputs, spentToday: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-500">Daily Limit (USDC)</span>
              <span className="font-mono text-purple-400">${inputs.dailyLimit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="1.00"
              step="0.05"
              value={inputs.dailyLimit}
              onChange={(e) => setInputs({ ...inputs, dailyLimit: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-gray-700 my-4" />

        {/* Outputs */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Decision</span>
            <span className={`text-sm font-semibold ${shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
              {shouldBuy ? 'APPROVE' : 'REJECT'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Confidence</span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${shouldBuy ? 'bg-green-500' : 'bg-red-500'}`}
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <span className="text-xs font-mono text-gray-300">{confidence}%</span>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Risk Score</span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${riskScore < 30 ? 'bg-green-500' : riskScore < 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${riskScore}%` }}
                />
              </div>
              <span className="text-xs font-mono text-gray-300">{riskScore}</span>
            </div>
          </div>
        </div>

        {/* Footer hint */}
        <div className="mt-4 pt-3 border-t border-gray-700">
          <p className="text-xs text-gray-500 text-center">
            Adjust inputs to see model output change
          </p>
        </div>
      </div>
    </div>
  );
}
