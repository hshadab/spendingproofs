/**
 * React Hook for Morpho Spending Proofs
 */

import { useState, useCallback, useRef } from 'react';
import type {
  SpendingPolicy,
  MorphoProofRequest,
  SpendingProof,
  GatedTxRequest,
  PreparedGatedTx,
  MorphoSpendingProofsConfig,
  Address,
} from '../types';
import { MorphoSpendingProofsClient, ProofStatus } from '../client';

/**
 * Hook state
 */
export interface UseMorphoSpendingProofState {
  /** Current status of proof generation */
  status: ProofStatus;
  /** Progress percentage (0-100) */
  progress: number;
  /** Generated proof (if complete) */
  proof: SpendingProof | null;
  /** Prepared transaction (if complete) */
  preparedTx: PreparedGatedTx | null;
  /** Error message (if failed) */
  error: string | null;
  /** Whether proof generation is in progress */
  isLoading: boolean;
}

/**
 * Hook actions
 */
export interface UseMorphoSpendingProofActions {
  /** Generate a proof for a Morpho operation */
  generateProof: (request: MorphoProofRequest) => Promise<SpendingProof | null>;
  /** Prepare a gated transaction with proof */
  prepareTransaction: (request: GatedTxRequest, policy: SpendingPolicy) => Promise<PreparedGatedTx | null>;
  /** Reset the hook state */
  reset: () => void;
  /** Clear any cached proofs */
  clearCache: () => void;
}

/**
 * Hook configuration
 */
export interface UseMorphoSpendingProofConfig extends MorphoSpendingProofsConfig {
  /** Signer for proof signatures */
  signer?: {
    signMessage: (message: string) => Promise<string>;
  };
}

/**
 * React hook for generating Morpho spending proofs
 *
 * @example
 * ```tsx
 * const { status, progress, proof, generateProof } = useMorphoSpendingProof({
 *   proverUrl: 'https://prover.example.com',
 *   chainId: 1,
 *   gateAddress: '0x...',
 *   morphoAddress: '0x...',
 *   signer: wallet,
 * });
 *
 * const handleSupply = async () => {
 *   const proof = await generateProof({
 *     policy: myPolicy,
 *     operation: MorphoOperation.SUPPLY,
 *     amount: parseUnits('1000', 6),
 *     market: marketAddress,
 *     agent: agentAddress,
 *   });
 *
 *   if (proof) {
 *     // Submit transaction with proof
 *   }
 * };
 * ```
 */
export function useMorphoSpendingProof(
  config: UseMorphoSpendingProofConfig,
): UseMorphoSpendingProofState & UseMorphoSpendingProofActions {
  const [state, setState] = useState<UseMorphoSpendingProofState>({
    status: 'idle',
    progress: 0,
    proof: null,
    preparedTx: null,
    error: null,
    isLoading: false,
  });

  const clientRef = useRef<MorphoSpendingProofsClient | null>(null);

  // Initialize client on first use
  const getClient = useCallback(() => {
    if (!clientRef.current) {
      clientRef.current = new MorphoSpendingProofsClient(config);
    }
    return clientRef.current;
  }, [config]);

  const handleProgress = useCallback((status: ProofStatus, progress: number) => {
    setState((prev) => ({
      ...prev,
      status,
      progress,
      isLoading: status !== 'complete' && status !== 'error' && status !== 'idle',
    }));
  }, []);

  const generateProof = useCallback(
    async (request: MorphoProofRequest): Promise<SpendingProof | null> => {
      if (!config.signer) {
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: 'No signer provided',
          isLoading: false,
        }));
        return null;
      }

      setState({
        status: 'preparing',
        progress: 0,
        proof: null,
        preparedTx: null,
        error: null,
        isLoading: true,
      });

      try {
        const client = getClient();
        const proof = await client.generateProof(request, config.signer, handleProgress);

        setState((prev) => ({
          ...prev,
          status: 'complete',
          progress: 100,
          proof,
          isLoading: false,
        }));

        return proof;
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Unknown error';
        setState((prev) => ({
          ...prev,
          status: 'error',
          error,
          isLoading: false,
        }));
        return null;
      }
    },
    [config.signer, getClient, handleProgress],
  );

  const prepareTransaction = useCallback(
    async (request: GatedTxRequest, policy: SpendingPolicy): Promise<PreparedGatedTx | null> => {
      if (!config.signer) {
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: 'No signer provided',
          isLoading: false,
        }));
        return null;
      }

      setState({
        status: 'preparing',
        progress: 0,
        proof: null,
        preparedTx: null,
        error: null,
        isLoading: true,
      });

      try {
        const client = getClient();
        const preparedTx = await client.prepareGatedTransaction(
          request,
          policy,
          config.signer,
          handleProgress,
        );

        setState((prev) => ({
          ...prev,
          status: 'complete',
          progress: 100,
          proof: preparedTx.proof,
          preparedTx,
          isLoading: false,
        }));

        return preparedTx;
      } catch (err) {
        const error = err instanceof Error ? err.message : 'Unknown error';
        setState((prev) => ({
          ...prev,
          status: 'error',
          error,
          isLoading: false,
        }));
        return null;
      }
    },
    [config.signer, getClient, handleProgress],
  );

  const reset = useCallback(() => {
    setState({
      status: 'idle',
      progress: 0,
      proof: null,
      preparedTx: null,
      error: null,
      isLoading: false,
    });
  }, []);

  const clearCache = useCallback(() => {
    clientRef.current = null;
  }, []);

  return {
    ...state,
    generateProof,
    prepareTransaction,
    reset,
    clearCache,
  };
}
