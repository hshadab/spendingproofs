/**
 * Daily Spend Policy Template
 *
 * Simple time-based spending limits:
 * - Rolling daily limits
 * - Hourly limits
 * - Weekly aggregates
 */

import type { SpendingPolicy, Address } from '../types';

/**
 * Time-based spending configuration
 */
export interface DailySpendConfig {
  /** Daily limit in USD */
  dailyLimitUSD: number;
  /** Optional hourly limit (for burst protection) */
  hourlyLimitUSD?: number;
  /** Allowed markets */
  allowedMarkets?: Address[];
  /** Token decimals */
  tokenDecimals?: number;
  /** Which operations require proof */
  operationsRequiringProof?: {
    supply?: boolean;
    borrow?: boolean;
    withdraw?: boolean;
  };
}

/**
 * Spending schedule for time-of-day restrictions
 */
export interface SpendingSchedule {
  /** Allowed hours in UTC (0-23) */
  allowedHours: number[];
  /** Allowed days (0=Sunday, 6=Saturday) */
  allowedDays: number[];
}

/**
 * Create a daily spend policy
 */
export function createDailySpendPolicy(config: DailySpendConfig): SpendingPolicy {
  const decimals = config.tokenDecimals ?? 6;
  const multiplier = BigInt(10 ** decimals);

  const ops = config.operationsRequiringProof ?? {
    supply: true,
    borrow: true,
    withdraw: true,
  };

  return {
    dailyLimit: BigInt(config.dailyLimitUSD) * multiplier,
    maxSingleTx: config.hourlyLimitUSD
      ? BigInt(config.hourlyLimitUSD) * multiplier
      : BigInt(0),
    maxLTV: 10000, // No LTV restriction
    minHealthFactor: 0, // No health factor restriction
    allowedMarkets: config.allowedMarkets ?? [],
    requireProofForSupply: ops.supply ?? true,
    requireProofForBorrow: ops.borrow ?? true,
    requireProofForWithdraw: ops.withdraw ?? true,
  };
}

/**
 * Calculate remaining daily allowance
 */
export function calculateRemainingAllowance(
  dailyLimit: bigint,
  spent: bigint,
  lastResetTimestamp: number,
): { remaining: bigint; resetsAt: number } {
  const now = Math.floor(Date.now() / 1000);
  const dayInSeconds = 86400;
  const nextReset = lastResetTimestamp + dayInSeconds;

  if (now >= nextReset) {
    // Daily limit has reset
    return {
      remaining: dailyLimit,
      resetsAt: now + dayInSeconds,
    };
  }

  const remaining = dailyLimit > spent ? dailyLimit - spent : BigInt(0);
  return {
    remaining,
    resetsAt: nextReset,
  };
}

/**
 * Check if current time is within allowed schedule
 */
export function isWithinSchedule(schedule: SpendingSchedule): boolean {
  const now = new Date();
  const hour = now.getUTCHours();
  const day = now.getUTCDay();

  return schedule.allowedHours.includes(hour) && schedule.allowedDays.includes(day);
}

/**
 * Create a business hours schedule (Mon-Fri, 9am-5pm UTC)
 */
export function createBusinessHoursSchedule(): SpendingSchedule {
  return {
    allowedHours: [9, 10, 11, 12, 13, 14, 15, 16, 17],
    allowedDays: [1, 2, 3, 4, 5], // Monday to Friday
  };
}

/**
 * Create a 24/7 schedule
 */
export function create24x7Schedule(): SpendingSchedule {
  return {
    allowedHours: Array.from({ length: 24 }, (_, i) => i),
    allowedDays: [0, 1, 2, 3, 4, 5, 6],
  };
}

/**
 * Format remaining allowance for display
 */
export function formatAllowance(
  remaining: bigint,
  decimals: number = 6,
  symbol: string = 'USDC',
): string {
  const divisor = BigInt(10 ** decimals);
  const whole = remaining / divisor;
  const fraction = remaining % divisor;

  const fractionStr = fraction.toString().padStart(decimals, '0').slice(0, 2);
  return `${whole.toLocaleString()}.${fractionStr} ${symbol}`;
}

/**
 * Validate daily spend configuration
 */
export function validateDailySpendConfig(config: DailySpendConfig): {
  valid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (config.dailyLimitUSD <= 0) {
    errors.push('Daily limit must be positive');
  }

  if (config.hourlyLimitUSD) {
    if (config.hourlyLimitUSD <= 0) {
      errors.push('Hourly limit must be positive');
    }
    if (config.hourlyLimitUSD > config.dailyLimitUSD) {
      warnings.push('Hourly limit exceeds daily limit - hourly limit is ineffective');
    }
    if (config.hourlyLimitUSD * 24 < config.dailyLimitUSD) {
      warnings.push('Cannot spend full daily limit with current hourly restrictions');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
