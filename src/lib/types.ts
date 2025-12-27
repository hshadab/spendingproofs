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
 * Unified spending proof representation for enforcement demos
 */
export interface SpendingProof {
  proofHash: string;
  inputHash: string;
  modelHash: string;
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
