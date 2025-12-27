'use client';

import { useState, useCallback, useRef } from 'react';
import type { ProofGenerationState, ProveResponse, ProofStep } from '@/lib/types';
import { spendingInputToNumeric, type SpendingModelInput } from '@/lib/spendingModel';

const INITIAL_STATE: ProofGenerationState = {
  status: 'idle',
  progress: 0,
  currentStep: '',
  elapsedMs: 0,
  steps: [],
};

export function useProofGeneration() {
  const [state, setState] = useState<ProofGenerationState>(INITIAL_STATE);
  const startTimeRef = useRef<number>(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const generateProof = useCallback(async (input: SpendingModelInput): Promise<ProveResponse> => {
    // Reset state
    setState({
      status: 'running',
      progress: 0,
      currentStep: 'Preparing inputs',
      elapsedMs: 0,
      steps: [
        { name: 'Running ONNX inference', status: 'pending' },
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
      const stepProgress = Math.floor((progress / 100) * 5);
      const steps: ProofStep[] = [
        { name: 'Running ONNX inference', status: stepProgress > 0 ? 'done' : stepProgress === 0 ? 'running' : 'pending', durationMs: stepProgress > 0 ? 50 : undefined },
        { name: 'Preparing witness data', status: stepProgress > 1 ? 'done' : stepProgress === 1 ? 'running' : 'pending', durationMs: stepProgress > 1 ? 200 : undefined },
        { name: 'Generating JOLT SNARK proof', status: stepProgress > 2 ? 'done' : stepProgress === 2 ? 'running' : 'pending', durationMs: stepProgress > 2 ? 5000 : undefined },
        { name: 'Computing commitments', status: stepProgress > 3 ? 'done' : stepProgress === 3 ? 'running' : 'pending', durationMs: stepProgress > 3 ? 1500 : undefined },
        { name: 'Finalizing proof', status: stepProgress > 4 ? 'done' : stepProgress === 4 ? 'running' : 'pending' },
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

      // Call the prove API
      const response = await fetch('/api/prove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: numericInputs, tag: 'spending' }),
      });

      const result: ProveResponse = await response.json();

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
            { name: 'Preparing witness data', status: 'done', durationMs: 200 },
            { name: 'Generating JOLT SNARK proof', status: 'done', durationMs: result.generationTimeMs - 1750 },
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
