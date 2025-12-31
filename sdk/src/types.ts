/**
 * Arc Policy Proofs SDK - Type Definitions
 */

/**
 * Spending policy configuration
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
 * Input for spending decision model
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
 * Arc Testnet configuration
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
