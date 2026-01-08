/**
 * Policy Templates for Morpho Spending Proofs
 */

export * from './vault-limits';
export * from './ltv-bounds';
export * from './daily-spend';

import type { SpendingPolicy } from '../types';
import { createVaultLimitsPolicy, VAULT_LIMITS_PRESETS } from './vault-limits';
import { createLTVBoundsPolicy, LTV_RISK_TIERS } from './ltv-bounds';
import { createDailySpendPolicy } from './daily-spend';

/**
 * Combined policy configuration
 */
export interface CombinedPolicyConfig {
  /** Spending limits */
  dailyLimitUSD: number;
  maxSingleTxUSD: number;
  /** Risk parameters */
  maxLTVPercent: number;
  minHealthFactor: number;
  /** Market restrictions */
  allowedMarkets?: `0x${string}`[];
  /** Token decimals */
  tokenDecimals?: number;
}

/**
 * Create a comprehensive policy combining vault limits and LTV bounds
 */
export function createCombinedPolicy(config: CombinedPolicyConfig): SpendingPolicy {
  const decimals = config.tokenDecimals ?? 6;
  const multiplier = BigInt(10 ** decimals);

  return {
    dailyLimit: BigInt(config.dailyLimitUSD) * multiplier,
    maxSingleTx: BigInt(config.maxSingleTxUSD) * multiplier,
    maxLTV: Math.round(config.maxLTVPercent * 100),
    minHealthFactor: Math.round(config.minHealthFactor * 10000),
    allowedMarkets: config.allowedMarkets ?? [],
    requireProofForSupply: true,
    requireProofForBorrow: true,
    requireProofForWithdraw: true,
  };
}

/**
 * Preset combined policies for common agent types
 */
export const AGENT_POLICY_PRESETS = {
  /** Conservative yield farming agent */
  conservativeYieldAgent: createCombinedPolicy({
    dailyLimitUSD: 10_000,
    maxSingleTxUSD: 5_000,
    maxLTVPercent: 50,
    minHealthFactor: 1.5,
  }),

  /** Moderate rebalancing agent */
  moderateRebalanceAgent: createCombinedPolicy({
    dailyLimitUSD: 50_000,
    maxSingleTxUSD: 25_000,
    maxLTVPercent: 65,
    minHealthFactor: 1.3,
  }),

  /** Aggressive trading agent */
  aggressiveTradingAgent: createCombinedPolicy({
    dailyLimitUSD: 100_000,
    maxSingleTxUSD: 50_000,
    maxLTVPercent: 80,
    minHealthFactor: 1.15,
  }),

  /** Institutional treasury agent */
  institutionalTreasuryAgent: createCombinedPolicy({
    dailyLimitUSD: 1_000_000,
    maxSingleTxUSD: 500_000,
    maxLTVPercent: 60,
    minHealthFactor: 1.4,
  }),
} as const;

/**
 * Compute policy hash (matches on-chain computation)
 */
export function computePolicyHash(policy: SpendingPolicy): `0x${string}` {
  // This should match the Solidity keccak256(abi.encode(...))
  // In production, use ethers or viem to compute this
  const encoder = new TextEncoder();
  const data = JSON.stringify({
    dailyLimit: policy.dailyLimit.toString(),
    maxSingleTx: policy.maxSingleTx.toString(),
    maxLTV: policy.maxLTV,
    minHealthFactor: policy.minHealthFactor,
    allowedMarkets: policy.allowedMarkets,
    requireProofForSupply: policy.requireProofForSupply,
    requireProofForBorrow: policy.requireProofForBorrow,
    requireProofForWithdraw: policy.requireProofForWithdraw,
  });

  // Placeholder - in production use actual keccak256
  return `0x${'0'.repeat(64)}` as `0x${string}`;
}

/**
 * Validate any policy configuration
 */
export function validatePolicy(policy: SpendingPolicy): {
  valid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Basic validation
  if (policy.dailyLimit <= BigInt(0)) {
    errors.push('Daily limit must be positive');
  }

  if (policy.maxLTV < 0 || policy.maxLTV > 10000) {
    errors.push('Max LTV must be between 0 and 10000 basis points');
  }

  if (policy.minHealthFactor < 0) {
    errors.push('Min health factor cannot be negative');
  }

  // Warnings
  if (policy.maxLTV > 9000) {
    warnings.push('Max LTV above 90% is very risky');
  }

  if (policy.minHealthFactor > 0 && policy.minHealthFactor < 10500) {
    warnings.push('Health factor below 1.05 leaves little liquidation buffer');
  }

  if (policy.allowedMarkets.length === 0) {
    warnings.push('No market restrictions - agent can operate on any market');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
