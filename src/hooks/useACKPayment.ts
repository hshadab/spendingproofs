'use client';

import { useCallback, useState } from 'react';
import {
  useWriteContract,
  useWaitForTransactionReceipt,
  useReadContract,
} from 'wagmi';
import { parseUnits } from 'viem';
import { ADDRESSES } from '@/lib/config';

type Address = `0x${string}`;

// ERC20 ABI (just transfer and balanceOf)
const ERC20_ABI = [
  {
    name: 'transfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'value', type: 'uint256' },
    ],
    outputs: [{ name: '', type: 'bool' }],
  },
  {
    name: 'balanceOf',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'account', type: 'address' }],
    outputs: [{ name: '', type: 'uint256' }],
  },
] as const;

export interface GatedTransferResult {
  success: boolean;
  transfer?: {
    txHash: string;
    attestationTxHash?: string;
    explorerUrl: string;
    attestationExplorerUrl?: string;
    to: string;
    amount: number;
    proofHash: string;
    verificationHash: string;
    verification: {
      decision: boolean;
      confidence: number;
      timestamp: number;
    };
    note: string;
  };
  steps?: { step: string; status: string; txHash?: string; hash?: string }[];
  error?: string;
}

export interface UseACKPaymentReturn {
  // Execute payment (direct mode - user wallet)
  executeTransfer: (recipient: Address, amount: string) => void;

  // Execute gated payment (API mode - SpendingGateWallet)
  // Attests verificationHash (not proofHash) - captures that verification passed
  executeGatedTransfer: (
    recipient: Address,
    amount: string,
    proofHash: string,
    verification: {
      decision: boolean;    // shouldBuy result
      confidence: number;   // 0-1
    },
    agentDid?: string
  ) => Promise<GatedTransferResult>;

  // State (direct mode)
  isPending: boolean;
  isConfirming: boolean;
  isConfirmed: boolean;
  error: Error | null;
  txHash: `0x${string}` | undefined;

  // State (gated mode)
  gatedResult: GatedTransferResult | null;
  isGatedPending: boolean;
  gatedError: string | null;

  // Balance
  balance: bigint | undefined;
  formattedBalance: string;

  // Reset
  reset: () => void;
}

/**
 * Hook for executing USDC payments on Arc Testnet
 *
 * Supports two modes:
 * 1. Direct transfer (executeTransfer) - User signs tx from their wallet
 * 2. Gated transfer (executeGatedTransfer) - API handles attestation + SpendingGateWallet transfer
 *
 * Option B Flow (Gated):
 * 1. Submit proof to ProofAttestation contract (required)
 * 2. Execute gatedTransfer via SpendingGateWallet (checks attestation on-chain)
 */
export function useACKPayment(userAddress: Address | undefined): UseACKPaymentReturn {
  // Gated transfer state
  const [gatedResult, setGatedResult] = useState<GatedTransferResult | null>(null);
  const [isGatedPending, setIsGatedPending] = useState(false);
  const [gatedError, setGatedError] = useState<string | null>(null);

  // Write contract hook (for direct mode)
  const {
    writeContract,
    data: txHash,
    isPending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  // Wait for transaction receipt (direct mode)
  const {
    isLoading: isConfirming,
    isSuccess: isConfirmed,
    error: receiptError,
  } = useWaitForTransactionReceipt({ hash: txHash });

  // Read user's USDC balance
  const { data: balance } = useReadContract({
    address: ADDRESSES.usdc,
    abi: ERC20_ABI,
    functionName: 'balanceOf',
    args: userAddress ? [userAddress] : undefined,
    query: { enabled: !!userAddress },
  });

  // Format balance for display
  const formattedBalance =
    balance !== undefined ? (Number(balance) / 1e6).toFixed(2) : '0.00';

  // Execute direct transfer (user wallet)
  const executeTransfer = useCallback(
    (recipient: Address, amount: string) => {
      if (!userAddress) {
        console.error('Wallet not connected');
        return;
      }

      // Parse amount to USDC units (6 decimals)
      const amountInUnits = parseUnits(amount, 6);

      writeContract({
        address: ADDRESSES.usdc,
        abi: ERC20_ABI,
        functionName: 'transfer',
        args: [recipient, amountInUnits],
      });
    },
    [userAddress, writeContract]
  );

  // Execute gated transfer via API (SpendingGateWallet)
  // Attests verificationHash = keccak256(proofHash, decision, confidence, timestamp)
  const executeGatedTransfer = useCallback(
    async (
      recipient: Address,
      amount: string,
      proofHash: string,
      verification: { decision: boolean; confidence: number },
      agentDid?: string
    ): Promise<GatedTransferResult> => {
      setIsGatedPending(true);
      setGatedError(null);
      setGatedResult(null);

      try {
        const response = await fetch('/api/ack/transfer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            to: recipient,
            amount: parseFloat(amount),
            proofHash,
            // Verification result - this is what gets attested
            decision: verification.decision,
            confidence: verification.confidence,
            verifiedAt: Math.floor(Date.now() / 1000),
            agentDid,
          }),
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          const errorMsg = data.error || 'Transfer failed';
          setGatedError(errorMsg);
          setGatedResult({ success: false, error: errorMsg, steps: data.steps });
          return { success: false, error: errorMsg, steps: data.steps };
        }

        setGatedResult({ success: true, transfer: data.transfer, steps: data.steps });
        return { success: true, transfer: data.transfer, steps: data.steps };
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setGatedError(errorMsg);
        setGatedResult({ success: false, error: errorMsg });
        return { success: false, error: errorMsg };
      } finally {
        setIsGatedPending(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    resetWrite();
    setGatedResult(null);
    setGatedError(null);
    setIsGatedPending(false);
  }, [resetWrite]);

  const error = writeError || receiptError || null;

  return {
    // Direct mode
    executeTransfer,
    isPending,
    isConfirming,
    isConfirmed,
    error,
    txHash,

    // Gated mode
    executeGatedTransfer,
    gatedResult,
    isGatedPending,
    gatedError,

    // Balance
    balance: balance as bigint | undefined,
    formattedBalance,

    // Reset
    reset,
  };
}
