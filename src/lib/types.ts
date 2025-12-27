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
