/**
 * Arc Policy Proofs SDK - Wagmi Integration
 *
 * Hooks for integrating spending proofs with wagmi-based applications.
 * Provides proof-gated transfer functionality.
 */

import { ARC_TESTNET } from './types';

/**
 * SpendingGate contract ABI (subset for SDK)
 */
export const SPENDING_GATE_ABI = [
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
] as const;

/**
 * ERC20 ABI for USDC interactions
 */
export const ERC20_ABI = [
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
] as const;

/**
 * Arc Testnet chain configuration for wagmi
 */
export const arcTestnetChain = {
  id: ARC_TESTNET.chainId,
  name: ARC_TESTNET.name,
  nativeCurrency: {
    name: 'Arc',
    symbol: 'ARC',
    decimals: 18,
  },
  rpcUrls: {
    default: {
      http: [ARC_TESTNET.rpcUrl],
    },
  },
  blockExplorers: {
    default: {
      name: 'ArcScan',
      url: ARC_TESTNET.explorerUrl,
    },
  },
  testnet: true,
} as const;

/**
 * Gated transfer parameters for wagmi hook
 */
export interface WagmiGatedTransferParams {
  spendingGateAddress: `0x${string}`;
  recipient: `0x${string}`;
  amountWei: bigint;
  proofHash: `0x${string}`;
  txIntentHash: `0x${string}`;
  nonce: bigint;
  expiry: bigint;
}

/**
 * Generate wagmi writeContract args for gated transfer
 *
 * @example
 * ```tsx
 * import { useWriteContract } from 'wagmi';
 * import { getGatedTransferArgs } from '@hshadab/spending-proofs/wagmi';
 *
 * function TransferButton({ params }: { params: WagmiGatedTransferParams }) {
 *   const { writeContract } = useWriteContract();
 *
 *   const handleTransfer = () => {
 *     writeContract(getGatedTransferArgs(params));
 *   };
 *
 *   return <button onClick={handleTransfer}>Execute Transfer</button>;
 * }
 * ```
 */
export function getGatedTransferArgs(params: WagmiGatedTransferParams) {
  return {
    address: params.spendingGateAddress,
    abi: SPENDING_GATE_ABI,
    functionName: 'gatedTransfer' as const,
    args: [
      params.recipient,
      params.amountWei,
      params.proofHash,
      params.txIntentHash,
      params.nonce,
      params.expiry,
    ] as const,
  };
}

/**
 * Generate wagmi readContract args for checking nonce
 */
export function getNonceCheckArgs(
  spendingGateAddress: `0x${string}`,
  sender: `0x${string}`,
  nonce: bigint
) {
  return {
    address: spendingGateAddress,
    abi: SPENDING_GATE_ABI,
    functionName: 'isNonceUsed' as const,
    args: [sender, nonce] as const,
  };
}

/**
 * Generate wagmi readContract args for checking proof
 */
export function getProofCheckArgs(
  spendingGateAddress: `0x${string}`,
  proofHash: `0x${string}`
) {
  return {
    address: spendingGateAddress,
    abi: SPENDING_GATE_ABI,
    functionName: 'isProofUsed' as const,
    args: [proofHash] as const,
  };
}

/**
 * Generate wagmi readContract args for USDC balance
 */
export function getBalanceArgs(usdcAddress: `0x${string}`, account: `0x${string}`) {
  return {
    address: usdcAddress,
    abi: ERC20_ABI,
    functionName: 'balanceOf' as const,
    args: [account] as const,
  };
}

/**
 * Get explorer URL for transaction
 */
export function getExplorerTxUrl(txHash: string): string {
  return `${ARC_TESTNET.explorerUrl}/tx/${txHash}`;
}

/**
 * Get explorer URL for address
 */
export function getExplorerAddressUrl(address: string): string {
  return `${ARC_TESTNET.explorerUrl}/address/${address}`;
}

/**
 * Format USDC amount (6 decimals) to display string
 */
export function formatUSDC(amountWei: bigint): string {
  const amount = Number(amountWei) / 1_000_000;
  return amount.toFixed(2);
}

/**
 * Parse USDC amount to wei (6 decimals)
 */
export function parseUSDC(amount: number | string): bigint {
  const num = typeof amount === 'string' ? parseFloat(amount) : amount;
  return BigInt(Math.floor(num * 1_000_000));
}

/**
 * Example wagmi hook usage
 *
 * @example
 * ```tsx
 * // In your React app with wagmi:
 * import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
 * import { useState, useCallback } from 'react';
 * import {
 *   getGatedTransferArgs,
 *   WagmiGatedTransferParams,
 *   getExplorerTxUrl,
 * } from '@hshadab/spending-proofs/wagmi';
 * import { SpendingProofsWallet } from '@hshadab/spending-proofs/wallet';
 *
 * export function useProofGatedTransfer(proverUrl: string, agentAddress: string) {
 *   const wallet = useMemo(
 *     () => new SpendingProofsWallet({ proverUrl, agentAddress }),
 *     [proverUrl, agentAddress]
 *   );
 *
 *   const { writeContract, data: hash, isPending, error } = useWriteContract();
 *   const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({ hash });
 *
 *   const executeGatedTransfer = useCallback(
 *     async (params: {
 *       spendingGateAddress: `0x${string}`;
 *       recipient: `0x${string}`;
 *       amountUsdc: number;
 *       input: SpendingInput;
 *     }) => {
 *       // Generate proof
 *       const result = await wallet.prepareGatedTransfer({
 *         recipient: params.recipient,
 *         amountUsdc: params.amountUsdc,
 *         input: params.input,
 *       });
 *
 *       if (!result.approved) {
 *         throw new Error('Spending policy rejected transaction');
 *       }
 *
 *       // Execute on-chain
 *       writeContract(getGatedTransferArgs({
 *         spendingGateAddress: params.spendingGateAddress,
 *         recipient: params.recipient,
 *         amountWei: result.txIntent.amount,
 *         proofHash: result.proof.proofHash as `0x${string}`,
 *         txIntentHash: result.txIntentHash as `0x${string}`,
 *         nonce: result.txIntent.nonce,
 *         expiry: BigInt(result.txIntent.expiry),
 *       }));
 *
 *       return result;
 *     },
 *     [wallet, writeContract]
 *   );
 *
 *   return {
 *     executeGatedTransfer,
 *     hash,
 *     isPending,
 *     isConfirming,
 *     isSuccess,
 *     error,
 *     explorerUrl: hash ? getExplorerTxUrl(hash) : undefined,
 *   };
 * }
 * ```
 */
