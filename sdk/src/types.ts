/**
 * Arc Policy Proofs SDK - Type Definitions
 *
 * This module contains all TypeScript type definitions for the SDK.
 * Types are designed to be compatible with both browser and Node.js environments.
 *
 * @packageDocumentation
 */

/**
 * Spending policy configuration that defines limits and thresholds
 * for agent spending decisions.
 *
 * @example
 * ```typescript
 * const policy: SpendingPolicy = {
 *   dailyLimitUsdc: 10.0,      // $10/day max
 *   maxSinglePurchaseUsdc: 1.0, // $1 per transaction
 *   minServiceSuccessRate: 0.9, // 90% reliability required
 *   minBudgetBuffer: 0.5,       // Keep $0.50 reserve
 * };
 * ```
 */
export interface SpendingPolicy {
  /** Maximum allowed spend per day in USDC */
  dailyLimitUsdc: number;
  /** Maximum single purchase amount in USDC */
  maxSinglePurchaseUsdc: number;
  /** Minimum required service success rate (0-1) */
  minServiceSuccessRate: number;
  /** Minimum budget buffer to maintain in USDC */
  minBudgetBuffer: number;
}

/**
 * Input for the spending decision model.
 *
 * This represents all the information needed to make a spending decision.
 * These values are fed into the ONNX model and used to generate a SNARK proof.
 *
 * @example
 * ```typescript
 * const input: SpendingInput = {
 *   priceUsdc: 0.05,           // Buying something for $0.05
 *   budgetUsdc: 10.0,          // Have $10 in treasury
 *   spentTodayUsdc: 0.50,      // Already spent $0.50 today
 *   dailyLimitUsdc: 5.0,       // Daily limit is $5
 *   serviceSuccessRate: 0.95,  // Service has 95% success rate
 *   serviceTotalCalls: 100,    // Used service 100 times before
 *   purchasesInCategory: 3,    // 3 purchases in this category today
 *   timeSinceLastPurchase: 2.5 // 2.5 hours since last purchase
 * };
 * ```
 */
export interface SpendingInput {
  /** Price of the item/service in USDC */
  priceUsdc: number;
  /** Current budget available in USDC */
  budgetUsdc: number;
  /** Amount already spent today in USDC */
  spentTodayUsdc: number;
  /** Daily spending limit in USDC */
  dailyLimitUsdc: number;
  /** Service success rate (0-1) */
  serviceSuccessRate: number;
  /** Total calls made to the service */
  serviceTotalCalls: number;
  /** Number of purchases in this category */
  purchasesInCategory: number;
  /** Hours since last purchase */
  timeSinceLastPurchase: number;
}

/**
 * Output from spending decision model
 */
export interface SpendingDecision {
  /** Whether the purchase should be approved */
  shouldBuy: boolean;
  /** Confidence level (0-1) */
  confidence: number;
  /** Risk score (0-1, lower is safer) */
  riskScore: number;
}

/**
 * Proof metadata
 */
export interface ProofMetadata {
  /** Hash of the model used */
  modelHash: string;
  /** Hash of the inputs */
  inputHash: string;
  /** Hash of the outputs */
  outputHash: string;
  /** Proof size in bytes */
  proofSize: number;
  /** Time taken to generate proof in milliseconds */
  generationTimeMs: number;
}

/**
 * Complete proof result
 */
export interface ProofResult {
  /** The SNARK proof (hex encoded) */
  proof: string;
  /** Hash of the proof */
  proofHash: string;
  /** Proof metadata */
  metadata: ProofMetadata;
  /** The spending decision */
  decision: SpendingDecision;
  /** Raw model outputs */
  rawOutputs: number[];
}

/**
 * Verification result
 */
export interface VerificationResult {
  /** Whether the proof is valid */
  valid: boolean;
  /** Verification message */
  message: string;
  /** Expected input hash (if verification failed due to tamper) */
  expectedInputHash?: string;
  /** Actual input hash from proof */
  actualInputHash?: string;
}

/**
 * Prover service health status
 */
export interface ProverHealth {
  /** Whether the prover is healthy */
  healthy: boolean;
  /** Available models */
  models: string[];
  /** Prover version */
  version?: string;
}

/**
 * SDK configuration options
 */
export interface PolicyProofsConfig {
  /** Base URL for the prover service */
  proverUrl: string;
  /** Request timeout in milliseconds (default: 120000) */
  timeout?: number;
  /** Custom fetch implementation */
  fetch?: typeof fetch;
}

/**
 * On-chain verification options
 */
export interface OnChainVerifyOptions {
  /** Contract address for proof attestation */
  contractAddress: `0x${string}`;
  /** Chain ID */
  chainId: number;
  /** RPC URL */
  rpcUrl: string;
}

/**
 * Arc Testnet configuration constants
 */
export const ARC_TESTNET = {
  chainId: 5042002,
  name: 'Arc Testnet',
  rpcUrl: 'https://rpc.testnet.arc.network',
  explorerUrl: 'https://testnet.arcscan.app',
  contracts: {
    proofAttestation: '0xBE9a5DF7C551324CB872584C6E5bF56799787952' as `0x${string}`,
    spendingGateWallet: '0x6A47D13593c00359a1c5Fc6f9716926aF184d138' as `0x${string}`,
    testnetUsdc: '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213' as `0x${string}`,
  },
} as const;

/**
 * Error codes for SDK operations
 */
export enum SDKErrorCode {
  /** Prover service is not available */
  PROVER_UNAVAILABLE = 'PROVER_UNAVAILABLE',
  /** Proof generation timed out */
  PROVER_TIMEOUT = 'PROVER_TIMEOUT',
  /** Proof generation failed */
  PROOF_GENERATION_FAILED = 'PROOF_GENERATION_FAILED',
  /** Input validation failed */
  INVALID_INPUT = 'INVALID_INPUT',
  /** Proof verification failed */
  VERIFICATION_FAILED = 'VERIFICATION_FAILED',
  /** Network request failed */
  NETWORK_ERROR = 'NETWORK_ERROR',
  /** Unknown error */
  UNKNOWN = 'UNKNOWN',
}

/**
 * SDK Error class with typed error codes
 *
 * @example
 * ```typescript
 * try {
 *   await client.prove(input);
 * } catch (error) {
 *   if (error instanceof SDKError) {
 *     console.log('Error code:', error.code);
 *     console.log('Message:', error.message);
 *   }
 * }
 * ```
 */
export class SDKError extends Error {
  /** Error code for programmatic handling */
  readonly code: SDKErrorCode;
  /** Original error that caused this error */
  readonly cause?: unknown;
  /** Additional context about the error */
  readonly context?: Record<string, unknown>;

  constructor(
    message: string,
    code: SDKErrorCode = SDKErrorCode.UNKNOWN,
    cause?: unknown,
    context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'SDKError';
    this.code = code;
    this.cause = cause;
    this.context = context;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /** Create a prover unavailable error */
  static proverUnavailable(cause?: unknown): SDKError {
    return new SDKError(
      'Prover service is unavailable',
      SDKErrorCode.PROVER_UNAVAILABLE,
      cause
    );
  }

  /** Create a timeout error */
  static timeout(timeoutMs: number): SDKError {
    return new SDKError(
      `Request timed out after ${timeoutMs}ms`,
      SDKErrorCode.PROVER_TIMEOUT,
      undefined,
      { timeoutMs }
    );
  }

  /** Create an invalid input error */
  static invalidInput(errors: string[]): SDKError {
    return new SDKError(
      `Invalid input: ${errors.join('; ')}`,
      SDKErrorCode.INVALID_INPUT,
      undefined,
      { errors }
    );
  }
}
