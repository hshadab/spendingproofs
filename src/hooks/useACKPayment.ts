'use client';

import { useCallback } from 'react';
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

export interface UseACKPaymentReturn {
  // Execute payment (live mode)
  executeTransfer: (recipient: Address, amount: string) => void;

  // State
  isPending: boolean;
  isConfirming: boolean;
  isConfirmed: boolean;
  error: Error | null;
  txHash: `0x${string}` | undefined;

  // Balance
  balance: bigint | undefined;
  formattedBalance: string;

  // Reset
  reset: () => void;
}

/**
 * Hook for executing real USDC payments on Arc Testnet
 * Used by ACK demo in Live Mode
 */
export function useACKPayment(userAddress: Address | undefined): UseACKPaymentReturn {
  // Write contract hook
  const {
    writeContract,
    data: txHash,
    isPending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  // Wait for transaction receipt
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

  // Execute transfer
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

  const reset = useCallback(() => {
    resetWrite();
  }, [resetWrite]);

  const error = writeError || receiptError || null;

  return {
    executeTransfer,
    isPending,
    isConfirming,
    isConfirmed,
    error,
    txHash,
    balance: balance as bigint | undefined,
    formattedBalance,
    reset,
  };
}
