/**
 * JOLT-Atlas Prover Client
 *
 * Client for interacting with the JOLT-Atlas zkML prover service.
 */

import type {
  ProveResponse,
  VerifyResponse,
  HealthResponse,
} from './types';
import { keccak256, encodePacked } from 'viem';

const JOLT_ATLAS_URL = process.env.NEXT_PUBLIC_JOLT_ATLAS_URL || 'http://localhost:3001';

/**
 * Check if the prover service is healthy
 */
export async function checkProverHealth(): Promise<HealthResponse> {
  const response = await fetch(`${JOLT_ATLAS_URL}/health`);
  if (!response.ok) {
    throw new Error(`Prover health check failed: ${response.status}`);
  }
  const data = await response.json();
  return {
    status: data.status,
    version: data.version,
    proofType: data.proof_type,
    modelsLoaded: data.models_loaded,
  };
}

/**
 * Generate a zkML proof for spending model inputs
 */
export async function generateProof(
  inputs: number[],
  tag: string = 'spending'
): Promise<ProveResponse> {
  const response = await fetch(`${JOLT_ATLAS_URL}/prove`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_id: 'spending-model',
      inputs,
      tag,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    return {
      success: false,
      error: `Prover error: ${response.status} - ${error}`,
      generationTimeMs: 0,
    };
  }

  const data = await response.json();

  if (!data.success) {
    return {
      success: false,
      error: data.error || 'Proof generation failed',
      generationTimeMs: data.generation_time_ms || 0,
    };
  }

  return {
    success: true,
    proof: {
      proof: data.proof.proof,
      proofHash: data.proof.proof_hash,
      programIo: data.proof.program_io, // For SNARK verification
      metadata: {
        modelHash: data.proof.metadata.model_hash,
        inputHash: data.proof.metadata.input_hash,
        outputHash: data.proof.metadata.output_hash,
        proofSize: data.proof.metadata.proof_size,
        generationTime: data.proof.metadata.generation_time,
        proverVersion: data.proof.metadata.prover_version,
      },
      tag: data.proof.tag,
      timestamp: data.proof.timestamp,
    },
    inference: data.inference ? {
      output: data.inference.output,
      rawOutput: data.inference.raw_output,
      decision: data.inference.decision as 'approve' | 'reject',
      confidence: data.inference.confidence,
    } : undefined,
    generationTimeMs: data.generation_time_ms,
  };
}

/**
 * Verify a SNARK proof cryptographically
 * Requires program_io for real verification
 */
export async function verifyProof(
  proof: string,
  modelId: string,
  modelHash: string,
  programIo: string
): Promise<VerifyResponse> {
  const response = await fetch(`${JOLT_ATLAS_URL}/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      proof,
      model_id: modelId,
      model_hash: modelHash,
      program_io: programIo,
    }),
  });

  if (!response.ok) {
    return {
      valid: false,
      error: `Verify error: ${response.status}`,
      verificationTimeMs: 0,
    };
  }

  const data = await response.json();
  return {
    valid: data.valid,
    error: data.error,
    verificationTimeMs: data.verification_time_ms,
  };
}

/**
 * Compute keccak256 hash of input array (for local verification)
 * Uses real keccak256 for cryptographic security
 */
export function hashInputs(inputs: number[]): string {
  // Convert to fixed-point representation for deterministic hashing
  // Scale by 1e8 to preserve 8 decimal places, then convert to hex
  const scaledInputs = inputs.map(n => {
    const scaled = BigInt(Math.round(n * 1e8));
    return scaled;
  });

  // Pack as int256 values and hash
  const encoded = encodePacked(
    scaledInputs.map(() => 'int256' as const),
    scaledInputs
  );

  return keccak256(encoded);
}

/**
 * Verify proof locally by comparing input hashes
 */
export function verifyInputsMatch(
  originalInputs: number[],
  modifiedInputs: number[],
  proofInputHash: string
): { valid: boolean; reason: string; computedHash: string } {
  const originalHash = hashInputs(originalInputs);
  const modifiedHash = hashInputs(modifiedInputs);

  // Check if original matches proof
  const originalMatches = originalHash === proofInputHash;

  // Check if modified matches proof
  const modifiedMatches = modifiedHash === proofInputHash;

  if (modifiedMatches) {
    return {
      valid: true,
      reason: 'Input hash matches proof',
      computedHash: modifiedHash,
    };
  }

  if (originalMatches && !modifiedMatches) {
    return {
      valid: false,
      reason: 'Input hash mismatch - inputs were modified after proof generation',
      computedHash: modifiedHash,
    };
  }

  return {
    valid: false,
    reason: 'Input hash does not match proof',
    computedHash: modifiedHash,
  };
}
