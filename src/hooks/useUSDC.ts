'use client';

import { useReadContract, useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { parseUnits, formatUnits } from 'viem';
import { CONTRACTS, getExplorerUrl } from '@/lib/wagmi';
import { useCallback, useMemo } from 'react';

// Standard ERC20 ABI for USDC
const ERC20_ABI = [
  {
    name: 'transfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'amount', type: 'uint256' },
    ],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'approve',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'spender', type: 'address' },
      { name: 'amount', type: 'uint256' },
    ],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'balanceOf',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'account', type: 'address' }],
    outputs: [{ type: 'uint256' }],
  },
  {
    name: 'allowance',
    type: 'function',
    stateMutability: 'view',
    inputs: [
      { name: 'owner', type: 'address' },
      { name: 'spender', type: 'address' },
    ],
    outputs: [{ type: 'uint256' }],
  },
  {
    name: 'decimals',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ type: 'uint8' }],
  },
  {
    name: 'symbol',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ type: 'string' }],
  },
] as const;

// USDC has 6 decimals
const USDC_DECIMALS = 6;

export interface TransferParams {
  to: `0x${string}`;
  amount: number; // Amount in USDC (e.g., 0.05 for 5 cents)
}

export interface ApproveParams {
  spender: `0x${string}`;
  amount: number; // Amount in USDC
}

export function useUSDC() {
  const usdcAddress = CONTRACTS.usdc;
  const isConfigured = usdcAddress !== '0x0000000000000000000000000000000000000000';

  // Write contract hooks
  const {
    writeContract,
    data: hash,
    isPending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  // Wait for transaction receipt
  const {
    isLoading: isConfirming,
    isSuccess,
    data: receipt,
    error: receiptError,
  } = useWaitForTransactionReceipt({ hash });

  // Transfer USDC
  const transfer = useCallback(
    async (params: TransferParams) => {
      if (!isConfigured) {
        throw new Error('USDC address not configured');
      }

      const amountWei = parseUnits(params.amount.toString(), USDC_DECIMALS);

      writeContract({
        address: usdcAddress,
        abi: ERC20_ABI,
        functionName: 'transfer',
        args: [params.to, amountWei],
      });
    },
    [isConfigured, usdcAddress, writeContract]
  );

  // Approve USDC spending
  const approve = useCallback(
    async (params: ApproveParams) => {
      if (!isConfigured) {
        throw new Error('USDC address not configured');
      }

      const amountWei = parseUnits(params.amount.toString(), USDC_DECIMALS);

      writeContract({
        address: usdcAddress,
        abi: ERC20_ABI,
        functionName: 'approve',
        args: [params.spender, amountWei],
      });
    },
    [isConfigured, usdcAddress, writeContract]
  );

  // Computed values
  const explorerUrl = useMemo(() => {
    return hash ? getExplorerUrl('tx', hash) : undefined;
  }, [hash]);

  const error = writeError || receiptError;

  return {
    // State
    isConfigured,
    hash,
    isPending,
    isConfirming,
    isSuccess,
    receipt,
    error,
    explorerUrl,

    // Actions
    transfer,
    approve,
    reset: resetWrite,

    // Constants
    decimals: USDC_DECIMALS,
    address: usdcAddress,
  };
}

// Hook for reading USDC balance
export function useUSDCBalance(address: `0x${string}` | undefined) {
  const usdcAddress = CONTRACTS.usdc;
  const isConfigured = usdcAddress !== '0x0000000000000000000000000000000000000000';

  const { data, isLoading, error, refetch } = useReadContract({
    address: usdcAddress,
    abi: ERC20_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
    query: {
      enabled: isConfigured && !!address,
    },
  });

  const balance = useMemo(() => {
    if (!data) return undefined;
    return {
      raw: data as bigint,
      formatted: formatUnits(data as bigint, USDC_DECIMALS),
      display: `${formatUnits(data as bigint, USDC_DECIMALS)} USDC`,
    };
  }, [data]);

  return {
    balance,
    isLoading,
    error,
    refetch,
    isConfigured,
  };
}

// Hook for checking USDC allowance
export function useUSDCAllowance(
  owner: `0x${string}` | undefined,
  spender: `0x${string}` | undefined
) {
  const usdcAddress = CONTRACTS.usdc;
  const isConfigured = usdcAddress !== '0x0000000000000000000000000000000000000000';

  const { data, isLoading, error, refetch } = useReadContract({
    address: usdcAddress,
    abi: ERC20_ABI,
    functionName: 'allowance',
    args: owner && spender ? [owner, spender] : undefined,
    query: {
      enabled: isConfigured && !!owner && !!spender,
    },
  });

  const allowance = useMemo(() => {
    if (!data) return undefined;
    return {
      raw: data as bigint,
      formatted: formatUnits(data as bigint, USDC_DECIMALS),
    };
  }, [data]);

  return {
    allowance,
    isLoading,
    error,
    refetch,
    isConfigured,
  };
}

// Helper to format USDC amounts
export function formatUSDC(amount: bigint | number): string {
  if (typeof amount === 'bigint') {
    return formatUnits(amount, USDC_DECIMALS);
  }
  return amount.toFixed(USDC_DECIMALS);
}

// Helper to parse USDC amounts
export function parseUSDC(amount: string | number): bigint {
  return parseUnits(amount.toString(), USDC_DECIMALS);
}
