'use client';

import { useCallback, useMemo } from 'react';
import {
  useReadContract,
  useWriteContract,
  useWaitForTransactionReceipt,
} from 'wagmi';
import { SPENDING_GATE_WALLET_ABI } from '@/lib/contracts';
import { getExplorerUrl } from '@/lib/wagmi';

type Address = `0x${string}`;

export interface GatedTransferParams {
  to: Address;
  amount: bigint;
  proofHash: `0x${string}`;
  expiry: bigint;
}

export interface SpendingGateWalletState {
  balance: bigint | undefined;
  nonce: bigint | undefined;
  remainingDailyAllowance: bigint | undefined;
  dailyLimit: bigint | undefined;
  maxSingleTransfer: bigint | undefined;
  agentId: bigint | undefined;
}

/**
 * Hook for interacting with a SpendingGateWallet contract
 * Provides real on-chain enforcement of spending policies
 */
export function useSpendingGateWallet(walletAddress: Address | undefined) {
  // Read wallet state
  const { data: balance, refetch: refetchBalance } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'getBalance',
    query: { enabled: !!walletAddress },
  });

  const { data: nonce, refetch: refetchNonce } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'getCurrentNonce',
    query: { enabled: !!walletAddress },
  });

  const { data: remainingDailyAllowance, refetch: refetchAllowance } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'getRemainingDailyAllowance',
    query: { enabled: !!walletAddress },
  });

  const { data: dailyLimit } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'dailyLimit',
    query: { enabled: !!walletAddress },
  });

  const { data: maxSingleTransfer } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'maxSingleTransfer',
    query: { enabled: !!walletAddress },
  });

  const { data: agentId } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'agentId',
    query: { enabled: !!walletAddress },
  });

  // Write operations
  const {
    writeContract,
    data: txHash,
    isPending: isWritePending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  const {
    isLoading: isConfirming,
    isSuccess: isConfirmed,
    data: receipt,
    error: receiptError,
  } = useWaitForTransactionReceipt({ hash: txHash });

  // Check if proof is used
  const checkProofUsed = useCallback(
    async (proofHash: `0x${string}`) => {
      if (!walletAddress) return false;
      // This would need to be called separately as a read
      return false;
    },
    [walletAddress]
  );

  // Execute gated transfer
  const gatedTransfer = useCallback(
    (params: GatedTransferParams) => {
      if (!walletAddress) {
        throw new Error('Wallet address not set');
      }

      writeContract({
        address: walletAddress,
        abi: SPENDING_GATE_WALLET_ABI,
        functionName: 'gatedTransfer',
        args: [params.to, params.amount, params.proofHash, params.expiry],
      });
    },
    [walletAddress, writeContract]
  );

  // Deposit USDC
  const deposit = useCallback(
    (amount: bigint) => {
      if (!walletAddress) {
        throw new Error('Wallet address not set');
      }

      writeContract({
        address: walletAddress,
        abi: SPENDING_GATE_WALLET_ABI,
        functionName: 'deposit',
        args: [amount],
      });
    },
    [walletAddress, writeContract]
  );

  // Emergency withdraw
  const emergencyWithdraw = useCallback(
    (to: Address) => {
      if (!walletAddress) {
        throw new Error('Wallet address not set');
      }

      writeContract({
        address: walletAddress,
        abi: SPENDING_GATE_WALLET_ABI,
        functionName: 'emergencyWithdraw',
        args: [to],
      });
    },
    [walletAddress, writeContract]
  );

  // Refetch all state
  const refetch = useCallback(() => {
    refetchBalance();
    refetchNonce();
    refetchAllowance();
  }, [refetchBalance, refetchNonce, refetchAllowance]);

  // Explorer URL for transaction
  const explorerUrl = useMemo(() => {
    return txHash ? getExplorerUrl('tx', txHash) : undefined;
  }, [txHash]);

  const error = writeError || receiptError;

  const state: SpendingGateWalletState = {
    balance: balance as bigint | undefined,
    nonce: nonce as bigint | undefined,
    remainingDailyAllowance: remainingDailyAllowance as bigint | undefined,
    dailyLimit: dailyLimit as bigint | undefined,
    maxSingleTransfer: maxSingleTransfer as bigint | undefined,
    agentId: agentId as bigint | undefined,
  };

  return {
    // State
    state,
    txHash,
    isWritePending,
    isConfirming,
    isConfirmed,
    receipt,
    error,
    explorerUrl,

    // Actions
    gatedTransfer,
    deposit,
    emergencyWithdraw,
    checkProofUsed,
    refetch,
    reset: resetWrite,

    // Address
    walletAddress,
  };
}

/**
 * Hook to check if a proof has been used in a SpendingGateWallet
 */
export function useProofUsed(
  walletAddress: Address | undefined,
  proofHash: `0x${string}` | undefined
) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'isProofUsed',
    args: proofHash ? [proofHash] : undefined,
    query: { enabled: !!walletAddress && !!proofHash },
  });

  return {
    isUsed: data as boolean | undefined,
    isLoading,
    error,
    refetch,
  };
}

/**
 * Hook to compute transaction intent hash on-chain
 */
export function useTxIntentHash(
  walletAddress: Address | undefined,
  to: Address | undefined,
  amount: bigint | undefined,
  nonce: bigint | undefined,
  expiry: bigint | undefined
) {
  const { data, isLoading, error } = useReadContract({
    address: walletAddress,
    abi: SPENDING_GATE_WALLET_ABI,
    functionName: 'computeTxIntentHash',
    args: to && amount !== undefined && nonce !== undefined && expiry !== undefined
      ? [to, amount, nonce, expiry]
      : undefined,
    query: {
      enabled: !!walletAddress && !!to && amount !== undefined && nonce !== undefined && expiry !== undefined,
    },
  });

  return {
    txIntentHash: data as `0x${string}` | undefined,
    isLoading,
    error,
  };
}
