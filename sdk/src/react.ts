/**
 * Arc Policy Proofs SDK - React Hooks
 *
 * React hooks for spending proof generation and management.
 * For use in React applications.
 */

// Note: This file is provided as a reference implementation.
// In practice, you would import and use these patterns in your React app.

/**
 * Example hook implementation for React applications.
 *
 * @example
 * ```tsx
 * // In your React app:
 * import { useState, useCallback, useMemo } from 'react';
 * import { PolicyProofs, SpendingInput, ProofResult } from '@hshadab/spending-proofs';
 *
 * export function useSpendingProofs(proverUrl: string) {
 *   const client = useMemo(() => new PolicyProofs({ proverUrl }), [proverUrl]);
 *
 *   const [proof, setProof] = useState<ProofResult | null>(null);
 *   const [isLoading, setIsLoading] = useState(false);
 *   const [error, setError] = useState<Error | null>(null);
 *
 *   const generateProof = useCallback(async (input: SpendingInput) => {
 *     setIsLoading(true);
 *     setError(null);
 *     try {
 *       const result = await client.prove(input);
 *       setProof(result);
 *       return result;
 *     } catch (e) {
 *       setError(e instanceof Error ? e : new Error('Proof generation failed'));
 *       throw e;
 *     } finally {
 *       setIsLoading(false);
 *     }
 *   }, [client]);
 *
 *   const decide = useCallback((input: SpendingInput) => {
 *     return client.decide(input);
 *   }, [client]);
 *
 *   const reset = useCallback(() => {
 *     setProof(null);
 *     setError(null);
 *   }, []);
 *
 *   return {
 *     proof,
 *     isLoading,
 *     error,
 *     generateProof,
 *     decide,
 *     reset,
 *     client,
 *   };
 * }
 * ```
 */

// Export types for React usage
export type {
  SpendingInput,
  SpendingDecision,
  ProofResult,
  ProverHealth,
} from './types';

/**
 * Hook state type
 */
export interface UseSpendingProofsState {
  proof: import('./types').ProofResult | null;
  isLoading: boolean;
  error: Error | null;
}

/**
 * Hook actions type
 */
export interface UseSpendingProofsActions {
  generateProof: (input: import('./types').SpendingInput) => Promise<import('./types').ProofResult>;
  decide: (input: import('./types').SpendingInput) => import('./types').SpendingDecision;
  reset: () => void;
}

/**
 * Hook return type
 */
export interface UseSpendingProofsReturn extends UseSpendingProofsState, UseSpendingProofsActions {
  client: import('./client').PolicyProofs;
}

/**
 * React hook factory for creating spending proofs hook
 * This is provided for environments that can't import React directly
 *
 * @example
 * ```tsx
 * import { createUseSpendingProofs } from '@hshadab/spending-proofs/react';
 * import { useState, useCallback, useMemo } from 'react';
 *
 * const useSpendingProofs = createUseSpendingProofs({
 *   useState,
 *   useCallback,
 *   useMemo,
 * });
 *
 * function MyComponent() {
 *   const { proof, isLoading, generateProof } = useSpendingProofs('http://prover.url');
 *   // ...
 * }
 * ```
 */
export function createUseSpendingProofs(reactHooks: {
  useState: typeof import('react').useState;
  useCallback: typeof import('react').useCallback;
  useMemo: typeof import('react').useMemo;
}) {
  const { useState, useCallback, useMemo } = reactHooks;
  // Dynamic import to avoid bundling issues
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { PolicyProofs } = require('./client') as { PolicyProofs: typeof import('./client').PolicyProofs };
  type SpendingInput = import('./types').SpendingInput;
  type ProofResult = import('./types').ProofResult;

  return function useSpendingProofs(proverUrl: string): UseSpendingProofsReturn {
    const client = useMemo(() => new PolicyProofs({ proverUrl }), [proverUrl]);

    const [proof, setProof] = useState<ProofResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const generateProof = useCallback(
      async (input: SpendingInput): Promise<ProofResult> => {
        setIsLoading(true);
        setError(null);
        try {
          const result = await client.prove(input);
          setProof(result);
          return result;
        } catch (e) {
          const err = e instanceof Error ? e : new Error('Proof generation failed');
          setError(err);
          throw err;
        } finally {
          setIsLoading(false);
        }
      },
      [client]
    );

    const decide = useCallback(
      (input: SpendingInput) => {
        return client.decide(input);
      },
      [client]
    );

    const reset = useCallback(() => {
      setProof(null);
      setError(null);
    }, []);

    return {
      proof,
      isLoading,
      error,
      generateProof,
      decide,
      reset,
      client,
    };
  };
}
