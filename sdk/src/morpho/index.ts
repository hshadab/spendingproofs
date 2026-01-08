/**
 * Morpho Spending Proofs SDK
 *
 * zkML spending proofs for Morpho Blue integration
 *
 * @example
 * ```typescript
 * import {
 *   createMorphoSpendingProofsClient,
 *   createCombinedPolicy,
 *   MorphoOperation,
 * } from '@hshadab/morpho-spending-proofs';
 *
 * // Create policy
 * const policy = createCombinedPolicy({
 *   dailyLimitUSD: 10_000,
 *   maxSingleTxUSD: 5_000,
 *   maxLTVPercent: 70,
 *   minHealthFactor: 1.2,
 * });
 *
 * // Create client
 * const client = createMorphoSpendingProofsClient({
 *   proverUrl: 'https://prover.example.com',
 *   chainId: 1,
 *   gateAddress: '0x...',
 *   morphoAddress: '0x...',
 * });
 *
 * // Generate proof
 * const proof = await client.generateProof({
 *   policy,
 *   operation: MorphoOperation.SUPPLY,
 *   amount: 1000000000n, // 1000 USDC
 *   market: '0x...',
 *   agent: '0x...',
 * }, signer);
 * ```
 */

// Types
export * from './types';

// Client
export { MorphoSpendingProofsClient, createMorphoSpendingProofsClient } from './client';
export type { ProofStatus, ProofProgressCallback } from './client';

// Policies
export * from './policies';

// React Hooks
export * from './hooks';
