'use client';

import { useState, useCallback, useRef } from 'react';
import { keccak256, encodePacked, toBytes } from 'viem';
import type { ProofGenerationState, ProveResponse, ProofStep, TxIntent } from '@/lib/types';
import { spendingInputToNumeric, runSpendingModel, DEFAULT_SPENDING_POLICY, type SpendingModelInput } from '@/lib/spendingModel';
import { withRetry, proverRetryOptions } from '@/lib/retry';
import { ProverError, parseError } from '@/lib/errors';
import { generateMockProofHash, generateMockHashes } from '@/lib/crypto';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('hooks:useProofGeneration');

/**
 * Create the message to sign for authenticated proof requests
 */
export function createProveMessage(inputs: number[], tag: string, timestamp: number): string {
  const inputHash = keccak256(toBytes(JSON.stringify(inputs)));
  return `Spending Proofs Authentication\n\nAction: Generate proof\nTag: ${tag}\nInput Hash: ${inputHash}\nTimestamp: ${timestamp}`;
}

/**
 * Signature function type for signing proof requests
 */
export type SignProveRequest = (message: string) => Promise<`0x${string}`>;

// Generate txIntentHash for proof binding using real keccak256
function generateTxIntentHash(input: SpendingModelInput, txIntent?: Partial<TxIntent>): string {
  const chainId = BigInt(txIntent?.chainId || 5042002);
  const amount = BigInt(Math.floor(input.priceUsdc * 1e6)); // USDC has 6 decimals
  const recipient = (txIntent?.recipient || '0x8ba1f109551bD432803012645Ac136ddd64DBA72') as `0x${string}`;
  const nonce = BigInt(txIntent?.nonce || Date.now());
  const expiry = BigInt(txIntent?.expiry || Math.floor(Date.now() / 1000) + 3600);
  const policyId = txIntent?.policyId || 'default-spending-policy';

  const encoded = encodePacked(
    ['uint256', 'uint256', 'address', 'uint256', 'uint256', 'string'],
    [chainId, amount, recipient, nonce, expiry, policyId]
  );

  return keccak256(encoded);
}

// Generate mock proof data for static demo (when prover is unavailable)
// WARNING: Mock proofs are NOT cryptographically valid and cannot be verified
function generateMockProof(input: SpendingModelInput, txIntent?: Partial<TxIntent>): ProveResponse {
  const decision = runSpendingModel(input, DEFAULT_SPENDING_POLICY);
  // Use cryptographically secure random generation
  const mockHash = generateMockProofHash();
  const { inputHash, outputHash } = generateMockHashes();
  const modelHash = '0x7a8b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890';
  const txIntentHash = generateTxIntentHash(input, txIntent);
  // Use crypto for generation time jitter
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  const jitter = (bytes[0] | (bytes[1] << 8)) / 65535; // 0-1 range
  const generationTime = 2000 + jitter * 4000;

  // Mock program_io - not valid for real verification
  const mockProgramIo = '0x' + Buffer.from(JSON.stringify({
    inputs: [],
    outputs: [decision.shouldBuy ? 1 : 0],
    _mock: true,
  })).toString('hex');

  return {
    success: true,
    proof: {
      proof: 'mock_proof_' + mockHash.slice(2, 18),
      proofHash: mockHash,
      programIo: mockProgramIo, // Mock - not valid for verification
      metadata: {
        modelHash,
        inputHash,
        outputHash,
        proofSize: Math.floor(45000 + Math.random() * 10000),
        generationTime,
        proverVersion: 'jolt-atlas-MOCK-v0.0.0', // Clearly marked as mock
        txIntentHash,
      },
      tag: 'spending',
      timestamp: Date.now(),
    },
    inference: {
      output: decision.shouldBuy ? 1 : 0,
      rawOutput: [decision.shouldBuy ? 1 : 0, decision.confidence / 100, decision.riskScore / 100],
      decision: decision.shouldBuy ? 'approve' : 'reject',
      confidence: decision.confidence,
    },
    generationTimeMs: generationTime,
  };
}

const INITIAL_STATE: ProofGenerationState = {
  status: 'idle',
  progress: 0,
  currentStep: '',
  elapsedMs: 0,
  steps: [],
};

export interface ProofGenerationOptions {
  /** Wallet address for signed requests */
  address?: `0x${string}`;
  /** Function to sign the request message */
  signMessage?: SignProveRequest;
}

export function useProofGeneration() {
  const [state, setState] = useState<ProofGenerationState>(INITIAL_STATE);
  const startTimeRef = useRef<number>(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const generateProof = useCallback(async (
    input: SpendingModelInput,
    options?: ProofGenerationOptions
  ): Promise<ProveResponse> => {
    // Reset state
    setState({
      status: 'running',
      progress: 0,
      currentStep: 'Preparing inputs',
      elapsedMs: 0,
      steps: [
        { name: 'Running ONNX inference', status: 'pending' },
        { name: 'Computing txIntentHash', status: 'pending' },
        { name: 'Preparing witness data', status: 'pending' },
        { name: 'Generating JOLT SNARK proof', status: 'pending' },
        { name: 'Computing commitments', status: 'pending' },
        { name: 'Finalizing proof', status: 'pending' },
      ],
    });

    startTimeRef.current = Date.now();

    // Start progress simulation
    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      const estimatedTotal = 8000; // 8 seconds estimate
      const progress = Math.min(95, (elapsed / estimatedTotal) * 100);

      // Update steps based on progress
      const stepProgress = Math.floor((progress / 100) * 6);
      const steps: ProofStep[] = [
        { name: 'Running ONNX inference', status: stepProgress > 0 ? 'done' : stepProgress === 0 ? 'running' : 'pending', durationMs: stepProgress > 0 ? 50 : undefined },
        { name: 'Computing txIntentHash', status: stepProgress > 1 ? 'done' : stepProgress === 1 ? 'running' : 'pending', durationMs: stepProgress > 1 ? 25 : undefined },
        { name: 'Preparing witness data', status: stepProgress > 2 ? 'done' : stepProgress === 2 ? 'running' : 'pending', durationMs: stepProgress > 2 ? 200 : undefined },
        { name: 'Generating JOLT SNARK proof', status: stepProgress > 3 ? 'done' : stepProgress === 3 ? 'running' : 'pending', durationMs: stepProgress > 3 ? 5000 : undefined },
        { name: 'Computing commitments', status: stepProgress > 4 ? 'done' : stepProgress === 4 ? 'running' : 'pending', durationMs: stepProgress > 4 ? 1500 : undefined },
        { name: 'Finalizing proof', status: stepProgress > 5 ? 'done' : stepProgress === 5 ? 'running' : 'pending' },
      ];

      const currentStepIdx = steps.findIndex((s) => s.status === 'running');
      const currentStep = currentStepIdx >= 0 ? steps[currentStepIdx].name : 'Finalizing...';

      setState((prev) => ({
        ...prev,
        progress,
        elapsedMs: elapsed,
        currentStep,
        steps,
      }));
    }, 100);

    try {
      // Convert input to numeric array
      const numericInputs = spendingInputToNumeric(input);

      let result: ProveResponse;

      // Use direct prover URL if configured, otherwise use API proxy
      const proverUrl = process.env.NEXT_PUBLIC_PROVER_URL;
      const apiEndpoint = proverUrl ? `${proverUrl}/prove` : '/api/prove';

      try {
        // Build request body
        const tag = 'spending';
        let requestBody: Record<string, unknown> = { inputs: numericInputs, tag };

        // Add signature if signing is available
        if (options?.address && options?.signMessage) {
          const timestamp = Date.now();
          const message = createProveMessage(numericInputs, tag, timestamp);
          const signature = await options.signMessage(message);
          requestBody = {
            ...requestBody,
            address: options.address,
            timestamp,
            signature,
          };
        }

        // Try to call the prover with retry logic
        const retryResult = await withRetry(
          async () => {
            const response = await fetch(apiEndpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
              const errorText = await response.text().catch(() => 'Unknown error');
              throw new Error(`Prover error (${response.status}): ${errorText}`);
            }

            const data = await response.json();

            // Check if prover returned an error code
            if (!data.success && data.error?.includes('PROVER_UNAVAILABLE')) {
              throw ProverError.unavailable();
            }

            return data as ProveResponse;
          },
          {
            ...proverRetryOptions,
            onRetry: (error, attempt, delayMs) => {
              logger.warn('Prover request failed, retrying', {
                action: 'generate_proof',
                attempt,
                delayMs,
                error,
              });
              setState((prev) => ({
                ...prev,
                currentStep: `Retrying... (attempt ${attempt})`,
              }));
            },
          }
        );

        if (retryResult.success && retryResult.data) {
          result = retryResult.data;
        } else {
          // All retries failed, fall back to mock
          throw retryResult.error || new Error('Prover unavailable after retries');
        }
      } catch (err) {
        // Prover not available (e.g., static GitHub Pages deployment)
        // Fall back to mock proof with realistic timing
        const parsedError = parseError(err);
        logger.warn('Prover unavailable after retries, using mock proof generation', {
          action: 'generate_proof',
          error: parsedError.message,
        });
        // Use crypto for delay jitter
        const bytes = new Uint8Array(4);
        crypto.getRandomValues(bytes);
        const jitter = (bytes[0] | (bytes[1] << 8)) / 65535;
        await new Promise((resolve) => setTimeout(resolve, 3000 + jitter * 2000));
        result = generateMockProof(input);
        // Log mock proof warning
        logger.warn('Mock proof generated - not cryptographically valid', {
          action: 'generate_proof',
          isMock: true,
        });
      }

      // Clear interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      const finalElapsed = Date.now() - startTimeRef.current;

      if (result.success && result.proof) {
        setState({
          status: 'complete',
          progress: 100,
          currentStep: 'Complete',
          elapsedMs: finalElapsed,
          steps: [
            { name: 'Running ONNX inference', status: 'done', durationMs: 50 },
            { name: 'Computing txIntentHash', status: 'done', durationMs: 25 },
            { name: 'Preparing witness data', status: 'done', durationMs: 200 },
            { name: 'Generating JOLT SNARK proof', status: 'done', durationMs: result.generationTimeMs - 1775 },
            { name: 'Computing commitments', status: 'done', durationMs: 1000 },
            { name: 'Finalizing proof', status: 'done', durationMs: 500 },
          ],
          result,
        });
      } else {
        setState({
          status: 'error',
          progress: 0,
          currentStep: 'Failed',
          elapsedMs: finalElapsed,
          steps: [
            { name: 'Running ONNX inference', status: 'error' },
            { name: 'Computing txIntentHash', status: 'pending' },
            { name: 'Preparing witness data', status: 'pending' },
            { name: 'Generating JOLT SNARK proof', status: 'pending' },
            { name: 'Computing commitments', status: 'pending' },
            { name: 'Finalizing proof', status: 'pending' },
          ],
          error: result.error || 'Proof generation failed',
        });
      }

      return result;
    } catch (error) {
      // Clear interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const finalElapsed = Date.now() - startTimeRef.current;

      setState({
        status: 'error',
        progress: 0,
        currentStep: 'Failed',
        elapsedMs: finalElapsed,
        steps: [],
        error: errorMessage,
      });

      return {
        success: false,
        error: errorMessage,
        generationTimeMs: finalElapsed,
      };
    }
  }, []);

  const reset = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setState(INITIAL_STATE);
  }, []);

  return {
    state,
    generateProof,
    reset,
  };
}
