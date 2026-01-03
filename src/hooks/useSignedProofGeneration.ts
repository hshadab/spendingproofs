'use client';

import { useCallback } from 'react';
import { useAccount, useWalletClient } from 'wagmi';
import { useProofGeneration, type ProofGenerationOptions } from './useProofGeneration';
import type { ProveResponse } from '@/lib/types';
import type { SpendingModelInput } from '@/lib/spendingModel';

/**
 * Hook for generating proofs with automatic wallet signature authentication.
 *
 * This wraps useProofGeneration and automatically signs requests using the
 * connected wallet. If no wallet is connected, it falls back to unsigned requests.
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { generateSignedProof, state, isWalletConnected } = useSignedProofGeneration();
 *
 *   const handleProve = async () => {
 *     const result = await generateSignedProof(input);
 *     if (result.success) {
 *       console.log('Proof generated:', result.proof);
 *     }
 *   };
 *
 *   return (
 *     <button onClick={handleProve} disabled={!isWalletConnected}>
 *       Generate Proof
 *     </button>
 *   );
 * }
 * ```
 */
export function useSignedProofGeneration() {
  const { address, isConnected } = useAccount();
  const { data: walletClient } = useWalletClient();
  const { state, generateProof, reset } = useProofGeneration();

  const generateSignedProof = useCallback(
    async (input: SpendingModelInput): Promise<ProveResponse> => {
      const options: ProofGenerationOptions = {};

      // Add signing capability if wallet is connected
      if (address && walletClient) {
        options.address = address;
        options.signMessage = async (message: string) => {
          return walletClient.signMessage({ message });
        };
      }

      return generateProof(input, options);
    },
    [address, walletClient, generateProof]
  );

  return {
    state,
    generateSignedProof,
    generateProof, // Expose unsigned version too
    reset,
    isWalletConnected: isConnected && !!walletClient,
    walletAddress: address,
  };
}
