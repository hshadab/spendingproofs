/**
 * Type definitions for the Arc Policy Proofs demo
 */

export interface ProofMetadata {
  modelHash: string;
  inputHash: string;
  outputHash: string;
  proofSize: number;
  generationTime: number;
  proverVersion: string;
  txIntentHash?: string; // Binds proof to specific transaction intent
}

export interface ProofData {
  proof: string;
  proofHash: string;
  /** Serialized program_io for SNARK verification */
  programIo?: string;
  metadata: ProofMetadata;
  tag: string;
  timestamp: number;
}

export interface InferenceResult {
  output: number;
  rawOutput: number[];
  decision: 'approve' | 'reject';
  confidence: number;
}

export interface ProveResponse {
  success: boolean;
  proof?: ProofData;
  inference?: InferenceResult;
  error?: string;
  generationTimeMs: number;
}

export interface VerifyResponse {
  valid: boolean;
  error?: string;
  verificationTimeMs: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  proofType: string;
  modelsLoaded: number;
  verificationEnabled?: boolean;
}

export interface ProofStep {
  name: string;
  status: 'pending' | 'running' | 'done' | 'error';
  durationMs?: number;
}

export type ProofStatus = 'idle' | 'running' | 'complete' | 'error';

export interface ProofGenerationState {
  status: ProofStatus;
  progress: number;
  currentStep: string;
  elapsedMs: number;
  steps: ProofStep[];
  result?: ProveResponse;
  error?: string;
}

/**
 * Unified spending proof representation for enforcement
 */
export interface SpendingProof {
  /** Raw proof bytes (hex encoded) - optional for lightweight references */
  proof?: string;
  proofHash: string;
  inputHash: string;
  modelHash: string;
  /** Serialized program_io for SNARK verification */
  programIo?: string;
  decision: {
    shouldBuy: boolean;
    confidence: number;
    riskScore: number;
  };
  timestamp: number;
  proofSizeBytes: number;
  generationTimeMs: number;
  verified: boolean;
  txIntentHash?: string;
}

/**
 * Transaction intent structure for proof binding
 */
export interface TxIntent {
  chainId: number;
  usdcAddress: string;
  sender: string;
  recipient: string;
  amount: bigint;
  nonce: bigint;
  expiry: number;
  policyId: string;
  policyVersion: number;
}

/**
 * Signed request for authenticated proof generation
 */
export interface SignedProveRequest {
  inputs: number[];
  tag: string;
  address: `0x${string}`;
  timestamp: number;
  signature: `0x${string}`;
}

/**
 * Error codes for proof API
 */
export type ProveErrorCode =
  | 'INVALID_SIGNATURE'
  | 'SIGNATURE_EXPIRED'
  | 'ADDRESS_NOT_ALLOWED'
  | 'RATE_LIMITED'
  | 'PROVER_UNAVAILABLE'
  | 'PROOF_GENERATION_FAILED';

/**
 * Verification step for on-chain proof verification (Crossmint demo)
 */
export interface VerificationStep {
  step: string;
  status: 'success' | 'failed' | 'skipped';
  txHash?: string;
  details?: string;
  timeMs?: number;
}

/**
 * Transaction execution result (Crossmint demo)
 */
export interface TransactionResult {
  success: boolean;
  txHash?: string;
  transferId?: string;
  error?: string;
  chain?: string;
  amount?: string;
  recipient?: string;
  // On-chain verification fields
  verifiedOnChain?: boolean;
  attestationTxHash?: string;
  steps?: VerificationStep[];
  proofHash?: string;
  // Proof verification fields
  proofVerified?: boolean;
  method?: 'crossmint' | 'direct';
  integrationMode?: string;
  crossmintTransactionId?: string;
}
