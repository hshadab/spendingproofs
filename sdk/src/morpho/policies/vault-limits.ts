/**
 * Vault Limits Policy Template
 *
 * Defines spending limits for vault operations including:
 * - Daily spending caps
 * - Single transaction limits
 * - Market whitelisting
 */

import type { SpendingPolicy, Address } from '../types';

/**
 * Configuration for vault limits policy
 */
export interface VaultLimitsConfig {
  /** Daily limit in USD (will be converted to base units) */
  dailyLimitUSD: number;
  /** Max single transaction in USD */
  maxSingleTxUSD: number;
  /** Allowed market addresses */
  allowedMarkets?: Address[];
  /** Token decimals for conversion (default: 6 for USDC) */
  tokenDecimals?: number;
}

/**
 * Preset configurations for common use cases
 */
export const VAULT_LIMITS_PRESETS = {
  /** Conservative: $1K daily, $500 max single tx */
  conservative: {
    dailyLimitUSD: 1_000,
    maxSingleTxUSD: 500,
  },
  /** Moderate: $10K daily, $5K max single tx */
  moderate: {
    dailyLimitUSD: 10_000,
    maxSingleTxUSD: 5_000,
  },
  /** Aggressive: $100K daily, $50K max single tx */
  aggressive: {
    dailyLimitUSD: 100_000,
    maxSingleTxUSD: 50_000,
  },
  /** Institutional: $1M daily, $500K max single tx */
  institutional: {
    dailyLimitUSD: 1_000_000,
    maxSingleTxUSD: 500_000,
  },
} as const;

/**
 * Create a vault limits spending policy
 */
export function createVaultLimitsPolicy(config: VaultLimitsConfig): SpendingPolicy {
  const decimals = config.tokenDecimals ?? 6;
  const multiplier = BigInt(10 ** decimals);

  return {
    dailyLimit: BigInt(config.dailyLimitUSD) * multiplier,
    maxSingleTx: BigInt(config.maxSingleTxUSD) * multiplier,
    maxLTV: 10000, // No LTV restriction
    minHealthFactor: 0, // No health factor restriction
    allowedMarkets: config.allowedMarkets ?? [],
    requireProofForSupply: true,
    requireProofForBorrow: true,
    requireProofForWithdraw: true,
  };
}

/**
 * Create a policy from a preset
 */
export function createPolicyFromPreset(
  preset: keyof typeof VAULT_LIMITS_PRESETS,
  allowedMarkets?: Address[],
): SpendingPolicy {
  const config = VAULT_LIMITS_PRESETS[preset];
  return createVaultLimitsPolicy({
    ...config,
    allowedMarkets,
  });
}

/**
 * Validate vault limits configuration
 */
export function validateVaultLimitsConfig(config: VaultLimitsConfig): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (config.dailyLimitUSD <= 0) {
    errors.push('Daily limit must be positive');
  }

  if (config.maxSingleTxUSD <= 0) {
    errors.push('Max single tx must be positive');
  }

  if (config.maxSingleTxUSD > config.dailyLimitUSD) {
    errors.push('Max single tx cannot exceed daily limit');
  }

  if (config.tokenDecimals && (config.tokenDecimals < 0 || config.tokenDecimals > 18)) {
    errors.push('Token decimals must be between 0 and 18');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
