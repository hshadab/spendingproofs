/**
 * Centralized Constants
 *
 * Contains magic numbers and configuration values used across the codebase.
 */

/**
 * USDC Token Constants
 */
export const USDC_DECIMALS = 6;
export const USDC_SCALE_FACTOR = 1_000_000; // 10^6

/**
 * Proof Generation Constants
 */
export const PROOF_INPUT_SCALE = 1e8; // Scale factor for fixed-point input hashing

/**
 * Gas Estimation Constants
 */
export const DEFAULT_GAS_ESTIMATE = 85_000;
export const GAS_ESTIMATE_VARIANCE = 10_000;

/**
 * Convert USDC amount (as number) to wei (BigInt) with proper precision handling.
 * Uses string-based conversion to avoid floating-point precision issues.
 *
 * @param amountUsdc - Amount in USDC (e.g., 1.50 for $1.50)
 * @returns BigInt representation in wei (e.g., 1500000n for $1.50)
 */
export function usdcToWei(amountUsdc: number): bigint {
  // Convert to string with enough precision, then scale
  const amountStr = amountUsdc.toFixed(USDC_DECIMALS);
  const [intPart, decPart = ''] = amountStr.split('.');
  const paddedDec = decPart.padEnd(USDC_DECIMALS, '0').slice(0, USDC_DECIMALS);
  return BigInt(intPart + paddedDec);
}

/**
 * Convert wei (BigInt) to USDC amount (number).
 *
 * @param amountWei - Amount in wei (BigInt)
 * @returns Amount in USDC (number)
 */
export function weiToUsdc(amountWei: bigint): number {
  return Number(amountWei) / USDC_SCALE_FACTOR;
}
