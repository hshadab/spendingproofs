/**
 * Arc Policy Proofs SDK - Utilities
 */

import { SpendingInput, SpendingDecision } from './types';

/**
 * Convert spending input object to array of numbers for model
 */
export function spendingInputToArray(input: SpendingInput): number[] {
  return [
    input.priceUsdc,
    input.budgetUsdc,
    input.spentTodayUsdc,
    input.dailyLimitUsdc,
    input.serviceSuccessRate,
    input.serviceTotalCalls,
    input.purchasesInCategory,
    input.timeSinceLastPurchase,
  ];
}

/**
 * Create spending input from array
 */
export function arrayToSpendingInput(arr: number[]): SpendingInput {
  if (arr.length !== 8) {
    throw new Error('Input array must have exactly 8 elements');
  }

  return {
    priceUsdc: arr[0],
    budgetUsdc: arr[1],
    spentTodayUsdc: arr[2],
    dailyLimitUsdc: arr[3],
    serviceSuccessRate: arr[4],
    serviceTotalCalls: arr[5],
    purchasesInCategory: arr[6],
    timeSinceLastPurchase: arr[7],
  };
}

/**
 * Parse raw model outputs into spending decision
 */
export function parseDecision(rawOutputs: number[]): SpendingDecision {
  if (!rawOutputs || rawOutputs.length < 3) {
    return { shouldBuy: false, confidence: 0, riskScore: 1 };
  }

  const [shouldBuyRaw, confidence, riskScore] = rawOutputs;
  return {
    shouldBuy: shouldBuyRaw > 0.5,
    confidence: Math.max(0, Math.min(1, confidence)),
    riskScore: Math.max(0, Math.min(1, riskScore)),
  };
}

/**
 * Hash inputs for verification
 * Simple deterministic hash for client-side verification
 */
export function hashInputs(inputs: number[]): string {
  // Create a deterministic string representation
  const str = inputs.map((n) => n.toFixed(8)).join(',');

  // Simple hash function (for demo - use keccak256 in production)
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }

  return '0x' + Math.abs(hash).toString(16).padStart(8, '0');
}

/**
 * Format proof hash for display
 */
export function formatProofHash(hash: string, length = 8): string {
  if (!hash) return '';
  if (hash.length <= length * 2 + 2) return hash;
  return `${hash.slice(0, length + 2)}...${hash.slice(-length)}`;
}

/**
 * Get Arc explorer URL for transaction
 */
export function getExplorerTxUrl(
  txHash: string,
  explorerUrl = 'https://testnet.arcscan.app'
): string {
  return `${explorerUrl}/tx/${txHash}`;
}

/**
 * Get Arc explorer URL for address
 */
export function getExplorerAddressUrl(
  address: string,
  explorerUrl = 'https://testnet.arcscan.app'
): string {
  return `${explorerUrl}/address/${address}`;
}

/**
 * Validate spending input
 */
export function validateSpendingInput(input: SpendingInput): string[] {
  const errors: string[] = [];

  if (input.priceUsdc < 0) {
    errors.push('Price must be non-negative');
  }
  if (input.budgetUsdc < 0) {
    errors.push('Budget must be non-negative');
  }
  if (input.spentTodayUsdc < 0) {
    errors.push('Spent today must be non-negative');
  }
  if (input.dailyLimitUsdc <= 0) {
    errors.push('Daily limit must be positive');
  }
  if (input.serviceSuccessRate < 0 || input.serviceSuccessRate > 1) {
    errors.push('Service success rate must be between 0 and 1');
  }
  if (input.serviceTotalCalls < 0) {
    errors.push('Total calls must be non-negative');
  }
  if (input.purchasesInCategory < 0) {
    errors.push('Purchases in category must be non-negative');
  }
  if (input.timeSinceLastPurchase < 0) {
    errors.push('Time since last purchase must be non-negative');
  }

  return errors;
}

/**
 * Create a default spending policy
 */
export function defaultPolicy() {
  return {
    dailyLimitUsdc: 1.0,
    maxSinglePurchaseUsdc: 0.25,
    minServiceSuccessRate: 0.8,
    minBudgetBuffer: 0.1,
  };
}
