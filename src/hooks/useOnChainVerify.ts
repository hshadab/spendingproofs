'use client';

import { useState, useCallback } from 'react';
import { usePublicClient } from 'wagmi';
import { CONTRACTS, ARC_PROOF_ATTESTATION_ABI } from '@/lib/contracts';

interface VerifyResult {
  valid: boolean;
  reason: string;
  txHash?: string;
}

export function useOnChainVerify() {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerifyResult | null>(null);
  const publicClient = usePublicClient();

  const verifyOnChain = useCallback(
    async (proofHash: `0x${string}`): Promise<VerifyResult> => {
      setIsVerifying(true);
      setResult(null);

      try {
        if (!publicClient) {
          throw new Error('No client available');
        }

        // Call isProofHashValid on the contract
        const isValid = await publicClient.readContract({
          address: CONTRACTS.arcProofAttestation,
          abi: ARC_PROOF_ATTESTATION_ABI,
          functionName: 'isProofHashValid',
          args: [proofHash],
        });

        const verifyResult: VerifyResult = {
          valid: isValid as boolean,
          reason: isValid ? 'Proof verified on Arc Testnet' : 'Proof not found or invalid',
        };

        setResult(verifyResult);
        return verifyResult;
      } catch (error) {
        const errorResult: VerifyResult = {
          valid: false,
          reason: error instanceof Error ? error.message : 'Verification failed',
        };
        setResult(errorResult);
        return errorResult;
      } finally {
        setIsVerifying(false);
      }
    },
    [publicClient]
  );

  const verifyInputs = useCallback(
    async (
      inputs: number[],
      proofInputHash: string
    ): Promise<{ valid: boolean; reason: string; computedHash: string }> => {
      try {
        const response = await fetch('/api/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ inputs, proofInputHash }),
        });

        const data = await response.json();
        return {
          valid: data.valid,
          reason: data.reason || (data.valid ? 'Valid' : 'Invalid'),
          computedHash: data.computedHash,
        };
      } catch (error) {
        return {
          valid: false,
          reason: error instanceof Error ? error.message : 'Verification failed',
          computedHash: '',
        };
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setIsVerifying(false);
  }, []);

  return {
    isVerifying,
    result,
    verifyOnChain,
    verifyInputs,
    reset,
  };
}
