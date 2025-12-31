/**
 * Arc Policy Proofs SDK
 *
 * Generate and verify zkML SNARK proofs for autonomous agent spending decisions.
 *
 * @example
 * ```typescript
 * import { PolicyProofs, ARC_TESTNET } from '@hshadab/spending-proofs';
 *
 * const client = new PolicyProofs({
 *   proverUrl: 'http://localhost:3001'
 * });
 *
 * // Generate a proof
 * const result = await client.prove({
 *   priceUsdc: 0.05,
 *   budgetUsdc: 1.00,
 *   spentTodayUsdc: 0.20,
 *   dailyLimitUsdc: 0.50,
 *   serviceSuccessRate: 0.95,
 *   serviceTotalCalls: 100,
 *   purchasesInCategory: 5,
 *   timeSinceLastPurchase: 2.5,
 * });
 *
 * console.log(result.decision.shouldBuy); // true
 * console.log(result.proofHash); // 0x...
 *
 * // Verify on-chain
 * const isValid = await isProofValidOnChain(result.proofHash);
 * ```
 *
 * @packageDocumentation
 */

// Main client
export { PolicyProofs } from './client';

// Types
export type {
  SpendingPolicy,
  SpendingInput,
  SpendingDecision,
  ProofMetadata,
  ProofResult,
  VerificationResult,
  ProverHealth,
  PolicyProofsConfig,
  OnChainVerifyOptions,
} from './types';

// Constants
export { ARC_TESTNET } from './types';

// Utilities
export {
  spendingInputToArray,
  arrayToSpendingInput,
  parseDecision,
  hashInputs,
  formatProofHash,
  getExplorerTxUrl,
  getExplorerAddressUrl,
  validateSpendingInput,
  defaultPolicy,
} from './utils';

// On-chain attestation (note: attestation storage, not cryptographic verification)
export {
  PROOF_ATTESTATION_ABI,
  isProofAttested,
  getProofTimestamp,
  encodeSubmitProof,
  getProofAttestationContract,
} from './onchain';
