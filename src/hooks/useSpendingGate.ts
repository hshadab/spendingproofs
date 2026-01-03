'use client';

import { useWriteContract, useWaitForTransactionReceipt, useReadContract } from 'wagmi';
import { keccak256, encodePacked, parseUnits } from 'viem';
import { CONTRACTS, getExplorerUrl } from '@/lib/wagmi';
import { useCallback, useMemo } from 'react';
import type { TxIntent } from '@/lib/types';

// SpendingGate ABI
const SPENDING_GATE_ABI = [
  {
    name: 'gatedTransfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'recipient', type: 'address' },
      { name: 'amount', type: 'uint256' },
      { name: 'proofHash', type: 'bytes32' },
      { name: 'txIntentHash', type: 'bytes32' },
      { name: 'nonce', type: 'uint256' },
      { name: 'expiry', type: 'uint256' },
    ],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'isNonceUsed',
    type: 'function',
    stateMutability: 'view',
    inputs: [
      { name: 'sender', type: 'address' },
      { name: 'nonce', type: 'uint256' },
    ],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'isProofUsed',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'getPolicyRegistry',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ type: 'address' }],
  },
] as const;

// Compute txIntentHash from transaction intent
export function computeTxIntentHash(intent: TxIntent): `0x${string}` {
  return keccak256(
    encodePacked(
      ['uint256', 'address', 'address', 'address', 'uint256', 'uint256', 'uint256', 'string', 'uint256'],
      [
        BigInt(intent.chainId),
        intent.usdcAddress as `0x${string}`,
        intent.sender as `0x${string}`,
        intent.recipient as `0x${string}`,
        intent.amount,
        intent.nonce,
        BigInt(intent.expiry),
        intent.policyId,
        BigInt(intent.policyVersion),
      ]
    )
  );
}

// Create TxIntent from simple parameters
export function createTxIntent(params: {
  sender: `0x${string}`;
  recipient: `0x${string}`;
  amountUsdc: number;
  policyId?: string;
}): TxIntent {
  return {
    chainId: 5042002, // Arc Testnet
    usdcAddress: CONTRACTS.usdc,
    sender: params.sender,
    recipient: params.recipient,
    amount: parseUnits(params.amountUsdc.toString(), 6), // USDC has 6 decimals
    nonce: BigInt(Date.now()),
    expiry: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
    policyId: params.policyId || 'default-spending-policy',
    policyVersion: 1,
  };
}

export interface GatedTransferParams {
  recipient: `0x${string}`;
  amountUsdc: number;
  proofHash: `0x${string}`;
  txIntent: TxIntent;
}

export function useSpendingGate() {
  const spendingGateAddress = CONTRACTS.spendingGateWallet;
  const isConfigured = !!spendingGateAddress;

  // Write contract
  const {
    writeContract,
    data: hash,
    isPending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  // Wait for receipt
  const {
    isLoading: isConfirming,
    isSuccess,
    data: receipt,
    error: receiptError,
  } = useWaitForTransactionReceipt({ hash });

  // Execute gated transfer
  const gatedTransfer = useCallback(
    async (params: GatedTransferParams) => {
      if (!isConfigured || !spendingGateAddress) {
        throw new Error('SpendingGate address not configured');
      }

      const txIntentHash = computeTxIntentHash(params.txIntent);
      const amountWei = parseUnits(params.amountUsdc.toString(), 6);

      writeContract({
        address: spendingGateAddress,
        abi: SPENDING_GATE_ABI,
        functionName: 'gatedTransfer',
        args: [
          params.recipient,
          amountWei,
          params.proofHash,
          txIntentHash,
          params.txIntent.nonce,
          BigInt(params.txIntent.expiry),
        ],
      });
    },
    [isConfigured, spendingGateAddress, writeContract]
  );

  // Explorer URL
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
    gatedTransfer,
    reset: resetWrite,

    // Helpers
    computeTxIntentHash,
    createTxIntent,

    // Contract address
    contractAddress: spendingGateAddress,
  };
}

// Hook to check if a nonce has been used
export function useNonceUsed(
  sender: `0x${string}` | undefined,
  nonce: bigint | undefined
) {
  const spendingGateAddress = CONTRACTS.spendingGateWallet;

  const { data, isLoading, error, refetch } = useReadContract({
    address: spendingGateAddress,
    abi: SPENDING_GATE_ABI,
    functionName: 'isNonceUsed',
    args: sender && nonce !== undefined ? [sender, nonce] : undefined,
    query: {
      enabled: !!spendingGateAddress && !!sender && nonce !== undefined,
    },
  });

  return {
    isUsed: data as boolean | undefined,
    isLoading,
    error,
    refetch,
  };
}

// Hook to check if a proof has been used
export function useProofUsed(proofHash: `0x${string}` | undefined) {
  const spendingGateAddress = CONTRACTS.spendingGateWallet;

  const { data, isLoading, error, refetch } = useReadContract({
    address: spendingGateAddress,
    abi: SPENDING_GATE_ABI,
    functionName: 'isProofUsed',
    args: proofHash ? [proofHash] : undefined,
    query: {
      enabled: !!spendingGateAddress && !!proofHash,
    },
  });

  return {
    isUsed: data as boolean | undefined,
    isLoading,
    error,
    refetch,
  };
}
