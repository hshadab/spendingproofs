/**
 * OpenMind Demo Wallet Integration
 *
 * Real Arc testnet wallet for USDC payments
 */

import { createPublicClient, createWalletClient, http, formatUnits, parseUnits } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';
import { createLogger } from '@/lib/metrics';
import { ARC_CHAIN, ADDRESSES } from '@/lib/config';

const logger = createLogger('lib:openmind:wallet');

// Arc Testnet chain definition
const arcTestnet = {
  id: ARC_CHAIN.id,
  name: ARC_CHAIN.name,
  nativeCurrency: {
    name: 'USDC',
    symbol: 'USDC',
    decimals: 6,
  },
  rpcUrls: {
    default: { http: [ARC_CHAIN.rpcUrl] },
  },
  blockExplorers: {
    default: { name: 'ArcScan', url: ARC_CHAIN.explorerUrl },
  },
} as const;

// USDC ABI (minimal for balanceOf and transfer)
const USDC_ABI = [
  {
    name: 'balanceOf',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'account', type: 'address' }],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'transfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'amount', type: 'uint256' },
    ],
    outputs: [{ name: '', type: 'bool' }],
  },
  {
    name: 'decimals',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint8' }],
  },
] as const;

// Create public client for reading
const publicClient = createPublicClient({
  chain: arcTestnet,
  transport: http(ARC_CHAIN.rpcUrl),
});

/**
 * Get the OpenMind demo wallet address
 */
export function getWalletAddress(): `0x${string}` {
  return (process.env.NEXT_PUBLIC_OPENMIND_WALLET_ADDRESS || '0x0') as `0x${string}`;
}

/**
 * Get wallet client for signing transactions
 */
function getWalletClient() {
  const privateKey = process.env.OPENMIND_WALLET_PRIVATE_KEY;
  if (!privateKey) {
    throw new Error('OPENMIND_WALLET_PRIVATE_KEY not set');
  }

  const account = privateKeyToAccount(privateKey as `0x${string}`);

  return createWalletClient({
    account,
    chain: arcTestnet,
    transport: http(ARC_CHAIN.rpcUrl),
  });
}

/**
 * Get USDC balance for the OpenMind demo wallet
 * Arc testnet uses native USDC (18 decimals) as the native currency
 */
export async function getWalletBalance(): Promise<{
  balanceUsdc: number;
  balanceRaw: bigint;
  address: `0x${string}`;
}> {
  const address = getWalletAddress();

  try {
    // Arc testnet uses native USDC as the gas token (18 decimals)
    const balance = await publicClient.getBalance({ address });

    // Convert from 18 decimals to USDC amount
    const balanceUsdc = parseFloat(formatUnits(balance, 18));

    logger.info('Wallet balance fetched', {
      action: 'get_balance',
      address,
      balanceUsdc,
    });

    return {
      balanceUsdc,
      balanceRaw: balance,
      address,
    };
  } catch (error) {
    logger.error('Failed to fetch wallet balance', { action: 'get_balance', error });
    throw error;
  }
}

/**
 * Execute a native USDC transfer on Arc testnet
 * Arc uses native USDC (18 decimals) as the gas token
 */
export async function executeUsdcTransfer(
  to: `0x${string}`,
  amountUsdc: number
): Promise<{
  txHash: `0x${string}`;
  amountUsdc: number;
  to: `0x${string}`;
  explorerUrl: string;
}> {
  const walletClient = getWalletClient();
  // Arc uses 18 decimals for native USDC
  const amountRaw = parseUnits(amountUsdc.toString(), 18);

  logger.info('Executing native USDC transfer', {
    action: 'transfer',
    to,
    amountUsdc,
  });

  try {
    // Send native USDC (not ERC-20)
    const txHash = await walletClient.sendTransaction({
      to,
      value: amountRaw,
    });

    logger.info('USDC transfer submitted', {
      action: 'transfer',
      txHash,
      to,
      amountUsdc,
    });

    // Wait for confirmation
    const receipt = await publicClient.waitForTransactionReceipt({ hash: txHash });

    logger.info('USDC transfer confirmed', {
      action: 'transfer',
      txHash,
      blockNumber: receipt.blockNumber,
      status: receipt.status,
    });

    return {
      txHash,
      amountUsdc,
      to,
      explorerUrl: `${ARC_CHAIN.explorerUrl}/tx/${txHash}`,
    };
  } catch (error) {
    logger.error('USDC transfer failed', { action: 'transfer', error });
    throw error;
  }
}

/**
 * Simulate a transfer (for demo purposes without spending real USDC)
 */
export async function simulateUsdcTransfer(
  to: `0x${string}`,
  amountUsdc: number
): Promise<{
  success: boolean;
  amountUsdc: number;
  to: `0x${string}`;
  reason: string;
}> {
  try {
    const { balanceUsdc } = await getWalletBalance();

    if (balanceUsdc < amountUsdc) {
      return {
        success: false,
        amountUsdc,
        to,
        reason: `Insufficient balance: $${balanceUsdc.toFixed(2)} < $${amountUsdc.toFixed(4)}`,
      };
    }

    return {
      success: true,
      amountUsdc,
      to,
      reason: 'Transfer would succeed (simulated)',
    };
  } catch (error) {
    return {
      success: false,
      amountUsdc,
      to,
      reason: `Simulation failed: ${error}`,
    };
  }
}

/**
 * Get wallet info for display
 */
export async function getWalletInfo(): Promise<{
  address: `0x${string}`;
  balanceUsdc: number;
  network: string;
  explorerUrl: string;
}> {
  const address = getWalletAddress();
  let balanceUsdc = 0;

  try {
    const result = await getWalletBalance();
    balanceUsdc = result.balanceUsdc;
  } catch {
    // Balance fetch failed, use 0
  }

  return {
    address,
    balanceUsdc,
    network: ARC_CHAIN.name,
    explorerUrl: `${ARC_CHAIN.explorerUrl}/address/${address}`,
  };
}
