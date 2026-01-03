/**
 * Arc Policy Proofs SDK - Client
 *
 * This module provides the main `PolicyProofs` client for generating
 * and verifying zkML spending proofs.
 *
 * @example Basic usage
 * ```typescript
 * import { PolicyProofs } from '@hshadab/spending-proofs';
 *
 * // Initialize client
 * const client = new PolicyProofs({
 *   proverUrl: 'https://prover.example.com',
 *   timeout: 60000, // 1 minute timeout
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
 * console.log('Should buy:', result.decision.shouldBuy);
 * console.log('Proof hash:', result.proofHash);
 * ```
 *
 * @packageDocumentation
 */

import {
  PolicyProofsConfig,
  SpendingInput,
  SpendingDecision,
  ProofResult,
  VerificationResult,
  ProverHealth,
} from './types';
import { spendingInputToArray, hashInputs, parseDecision } from './utils';

const DEFAULT_TIMEOUT = 120_000; // 2 minutes for SNARK generation

// Internal API response types
interface ProveApiResponse {
  success: boolean;
  error?: string;
  proof: {
    proof: string;
    proof_hash: string;
    metadata?: {
      model_hash?: string;
      input_hash?: string;
      output_hash?: string;
    };
  };
  inference: {
    raw_output: number[];
  };
}

interface HealthApiResponse {
  models?: string[];
  version?: string;
}

/**
 * PolicyProofs client for generating and verifying zkML spending proofs
 *
 * @example
 * ```typescript
 * import { PolicyProofs } from '@hshadab/spending-proofs';
 *
 * const client = new PolicyProofs({
 *   proverUrl: 'http://localhost:3001'
 * });
 *
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
 * ```
 */
export class PolicyProofs {
  private readonly proverUrl: string;
  private readonly timeout: number;
  private readonly fetchFn: typeof fetch;

  constructor(config: PolicyProofsConfig) {
    this.proverUrl = config.proverUrl.replace(/\/$/, '');
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
    this.fetchFn = config.fetch ?? fetch;
  }

  /**
   * Generate a SNARK proof for a spending decision
   *
   * @param input - Spending decision inputs
   * @param tag - Optional tag for the proof (default: "spending")
   * @returns Proof result with decision and proof data
   */
  async prove(input: SpendingInput, tag = 'spending'): Promise<ProofResult> {
    const inputs = spendingInputToArray(input);
    const startTime = Date.now();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchFn(`${this.proverUrl}/prove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: 'spending-model',
          inputs,
          tag,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Prover error: ${error}`);
      }

      const data = (await response.json()) as ProveApiResponse;

      if (!data.success) {
        throw new Error(data.error || 'Proof generation failed');
      }

      const generationTimeMs = Date.now() - startTime;
      const decision = parseDecision(data.inference.raw_output);

      return {
        proof: data.proof.proof,
        proofHash: data.proof.proof_hash,
        metadata: {
          modelHash: data.proof.metadata?.model_hash || '',
          inputHash: data.proof.metadata?.input_hash || hashInputs(inputs),
          outputHash: data.proof.metadata?.output_hash || '',
          proofSize: data.proof.proof?.length || 0,
          generationTimeMs,
        },
        decision,
        rawOutputs: data.inference.raw_output,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Verify a proof against expected inputs
   *
   * @param proof - The proof to verify
   * @param inputs - The inputs to verify against
   * @returns Verification result
   */
  async verify(
    proof: ProofResult,
    inputs: SpendingInput
  ): Promise<VerificationResult> {
    const inputArray = spendingInputToArray(inputs);
    const expectedHash = hashInputs(inputArray);
    const actualHash = proof.metadata.inputHash;

    if (expectedHash !== actualHash) {
      return {
        valid: false,
        message: 'Input hash mismatch - inputs have been tampered with',
        expectedInputHash: expectedHash,
        actualInputHash: actualHash,
      };
    }

    return {
      valid: true,
      message: 'Proof is valid for the provided inputs',
    };
  }

  /**
   * Run spending decision model locally (no proof)
   * Useful for quick decisions without proof overhead
   *
   * @param input - Spending decision inputs
   * @returns Spending decision
   */
  decide(input: SpendingInput): SpendingDecision {
    const {
      priceUsdc,
      budgetUsdc,
      spentTodayUsdc,
      dailyLimitUsdc,
      serviceSuccessRate,
      serviceTotalCalls,
      purchasesInCategory,
      timeSinceLastPurchase,
    } = input;

    // Policy checks
    const withinBudget = priceUsdc <= budgetUsdc;
    const withinDailyLimit = spentTodayUsdc + priceUsdc <= dailyLimitUsdc;
    const serviceReliable = serviceSuccessRate >= 0.8 && serviceTotalCalls >= 10;
    const notOverspending = purchasesInCategory < 10 || timeSinceLastPurchase >= 1;

    // Calculate scores
    const budgetScore = withinBudget ? 1.0 : 0.0;
    const limitScore = withinDailyLimit ? 1.0 : 0.0;
    const serviceScore = serviceReliable ? 1.0 : 0.5;
    const frequencyScore = notOverspending ? 1.0 : 0.7;

    const confidence =
      (budgetScore + limitScore + serviceScore + frequencyScore) / 4;
    const shouldBuy = withinBudget && withinDailyLimit && confidence >= 0.7;
    const riskScore = 1 - confidence;

    return { shouldBuy, confidence, riskScore };
  }

  /**
   * Check prover service health
   *
   * @returns Prover health status
   */
  async health(): Promise<ProverHealth> {
    try {
      const response = await this.fetchFn(`${this.proverUrl}/health`, {
        method: 'GET',
      });

      if (!response.ok) {
        return { healthy: false, models: [] };
      }

      const data = (await response.json()) as HealthApiResponse;
      return {
        healthy: true,
        models: data.models || ['spending-model'],
        version: data.version,
      };
    } catch {
      return { healthy: false, models: [] };
    }
  }
}
