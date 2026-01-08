/**
 * LTV Bounds Policy Template
 *
 * Defines loan-to-value and health factor constraints:
 * - Maximum LTV ratio
 * - Minimum health factor
 * - Risk-based market restrictions
 */

import type { SpendingPolicy, Address, PositionState } from '../types';

/**
 * Configuration for LTV bounds policy
 */
export interface LTVBoundsConfig {
  /** Maximum LTV in percentage (e.g., 70 for 70%) */
  maxLTVPercent: number;
  /** Minimum health factor (e.g., 1.2) */
  minHealthFactor: number;
  /** Daily limit (optional, 0 for unlimited) */
  dailyLimitUSD?: number;
  /** Allowed markets */
  allowedMarkets?: Address[];
  /** Token decimals */
  tokenDecimals?: number;
}

/**
 * Risk tier configurations
 */
export const LTV_RISK_TIERS = {
  /** Ultra-safe: 50% LTV, 1.5 health factor */
  ultraSafe: {
    maxLTVPercent: 50,
    minHealthFactor: 1.5,
  },
  /** Safe: 65% LTV, 1.3 health factor */
  safe: {
    maxLTVPercent: 65,
    minHealthFactor: 1.3,
  },
  /** Moderate: 75% LTV, 1.2 health factor */
  moderate: {
    maxLTVPercent: 75,
    minHealthFactor: 1.2,
  },
  /** Aggressive: 85% LTV, 1.1 health factor */
  aggressive: {
    maxLTVPercent: 85,
    minHealthFactor: 1.1,
  },
} as const;

/**
 * Create an LTV bounds spending policy
 */
export function createLTVBoundsPolicy(config: LTVBoundsConfig): SpendingPolicy {
  const decimals = config.tokenDecimals ?? 6;
  const multiplier = BigInt(10 ** decimals);

  // Convert percentage to basis points
  const maxLTV = Math.round(config.maxLTVPercent * 100);
  const minHealthFactor = Math.round(config.minHealthFactor * 10000);

  return {
    dailyLimit: config.dailyLimitUSD
      ? BigInt(config.dailyLimitUSD) * multiplier
      : BigInt(2) ** BigInt(128) - BigInt(1), // Max uint128 for "unlimited"
    maxSingleTx: BigInt(0), // No single tx limit
    maxLTV,
    minHealthFactor,
    allowedMarkets: config.allowedMarkets ?? [],
    requireProofForSupply: false, // Supply doesn't affect LTV negatively
    requireProofForBorrow: true, // Borrow increases LTV
    requireProofForWithdraw: true, // Withdraw can increase LTV
  };
}

/**
 * Create a policy from a risk tier
 */
export function createPolicyFromRiskTier(
  tier: keyof typeof LTV_RISK_TIERS,
  options?: {
    dailyLimitUSD?: number;
    allowedMarkets?: Address[];
  },
): SpendingPolicy {
  const tierConfig = LTV_RISK_TIERS[tier];
  return createLTVBoundsPolicy({
    ...tierConfig,
    ...options,
  });
}

/**
 * Calculate projected LTV after a borrow operation
 */
export function calculateProjectedLTV(
  currentState: PositionState,
  borrowAmount: bigint,
  borrowTokenPriceUSD: bigint,
): number {
  const additionalBorrowUSD = (borrowAmount * borrowTokenPriceUSD) / BigInt(10 ** 18);
  const newBorrowValue = currentState.borrowValueUSD + additionalBorrowUSD;

  if (currentState.collateralValueUSD === BigInt(0)) {
    return newBorrowValue > BigInt(0) ? 10000 : 0; // 100% or 0%
  }

  // Return LTV in basis points
  return Number((newBorrowValue * BigInt(10000)) / currentState.collateralValueUSD);
}

/**
 * Calculate projected health factor after a withdraw operation
 */
export function calculateProjectedHealthFactor(
  currentState: PositionState,
  withdrawAmount: bigint,
  collateralTokenPriceUSD: bigint,
  liquidationThreshold: number, // in basis points
): number {
  const withdrawValueUSD = (withdrawAmount * collateralTokenPriceUSD) / BigInt(10 ** 18);
  const newCollateralValue = currentState.collateralValueUSD - withdrawValueUSD;

  if (currentState.borrowValueUSD === BigInt(0)) {
    return 100000; // Infinite health factor
  }

  // HF = (collateral * liquidation threshold) / borrow
  const numerator = (newCollateralValue * BigInt(liquidationThreshold)) / BigInt(10000);
  return Number((numerator * BigInt(10000)) / currentState.borrowValueUSD);
}

/**
 * Check if an operation would violate LTV bounds
 */
export function wouldViolateLTVBounds(
  policy: SpendingPolicy,
  currentState: PositionState,
  operation: 'borrow' | 'withdraw',
  amount: bigint,
  tokenPriceUSD: bigint,
  liquidationThreshold?: number,
): { wouldViolate: boolean; projectedValue: number; limit: number } {
  if (operation === 'borrow') {
    const projectedLTV = calculateProjectedLTV(currentState, amount, tokenPriceUSD);
    return {
      wouldViolate: projectedLTV > policy.maxLTV,
      projectedValue: projectedLTV,
      limit: policy.maxLTV,
    };
  } else {
    const projectedHF = calculateProjectedHealthFactor(
      currentState,
      amount,
      tokenPriceUSD,
      liquidationThreshold ?? 8500, // Default 85% liquidation threshold
    );
    return {
      wouldViolate: projectedHF < policy.minHealthFactor,
      projectedValue: projectedHF,
      limit: policy.minHealthFactor,
    };
  }
}

/**
 * Validate LTV bounds configuration
 */
export function validateLTVBoundsConfig(config: LTVBoundsConfig): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (config.maxLTVPercent <= 0 || config.maxLTVPercent > 100) {
    errors.push('Max LTV must be between 0 and 100 percent');
  }

  if (config.minHealthFactor < 1) {
    errors.push('Min health factor must be at least 1.0');
  }

  if (config.minHealthFactor > 10) {
    errors.push('Min health factor seems unreasonably high (>10)');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
