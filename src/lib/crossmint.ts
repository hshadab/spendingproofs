/**
 * Crossmint API Client
 *
 * Server-side client for Crossmint wallet and transfer operations.
 *
 * INTEGRATION MODES:
 * 1. "crossmint" - Full Crossmint-managed transfers (requires supported chain)
 * 2. "hybrid" - Crossmint wallet + direct on-chain transfer (for unsupported chains like Arc)
 *
 * Supported chains for Crossmint transfers: Base, Polygon, Arbitrum, Optimism, etc.
 * Arc testnet requires Crossmint sales enablement.
 */

import { createLogger } from './metrics';
import { API_CONFIG } from './config';

const logger = createLogger('lib:crossmint');

const CROSSMINT_API_URL = API_CONFIG.crossmintApiUrl;
const CROSSMINT_SERVER_KEY = process.env.CROSSMINT_SERVER_KEY;

// Integration mode: 'crossmint' for full Crossmint transfers, 'hybrid' for Crossmint wallet + direct transfer
export const INTEGRATION_MODE = process.env.CROSSMINT_INTEGRATION_MODE || 'hybrid';

if (!CROSSMINT_SERVER_KEY) {
  logger.warn('CROSSMINT_SERVER_KEY not set - Crossmint API calls will fail', {
    action: 'init',
  });
}

// API version for wallet endpoints (using latest)
const API_VERSION = API_CONFIG.crossmintApiVersion;

/**
 * Crossmint wallet types
 */
export interface CrossmintWallet {
  address: string;
  type: string;
  chain: string;
  linkedUser?: string;
  createdAt: string;
}

export interface WalletBalance {
  currency: string;
  amount: string;
  chain: string;
}

export interface TransferResult {
  id: string;
  status: 'pending' | 'success' | 'failed';
  txHash?: string;
  chain: string;
  amount: string;
  recipient: string;
}

/**
 * Create a new Crossmint wallet for the agent
 * Uses MPC wallet (custodial, Fireblocks-backed) which doesn't require adminSigner
 */
export async function createWallet(
  userId: string = 'zkml-demo-agent'
): Promise<CrossmintWallet> {
  const chain = process.env.NEXT_PUBLIC_CROSSMINT_CHAIN || 'base-sepolia';

  const response = await fetch(`${CROSSMINT_API_URL}/${API_VERSION}/wallets`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-KEY': CROSSMINT_SERVER_KEY!,
    },
    body: JSON.stringify({
      chainType: 'evm',
      type: 'smart',
      linkedUser: `userId:${userId}`,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create wallet: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return {
    address: data.address,
    type: data.type,
    chain: data.config?.chain || chain,
    linkedUser: data.linkedUser,
    createdAt: data.createdAt,
  };
}

/**
 * Get or create wallet for the demo agent
 * Returns the existing wallet if it exists (idempotent by userId)
 */
export async function getOrCreateAgentWallet(): Promise<CrossmintWallet> {
  // Crossmint API is idempotent - calling create with same userId returns existing wallet
  return createWallet('zkml-demo-agent');
}

/**
 * List all wallets
 */
export async function listWallets(): Promise<CrossmintWallet[]> {
  const response = await fetch(`${CROSSMINT_API_URL}/${API_VERSION}/wallets`, {
    headers: {
      'X-API-KEY': CROSSMINT_SERVER_KEY!,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list wallets: ${response.status}`);
  }

  const data = await response.json();
  return data.map((w: Record<string, unknown>) => ({
    address: w.address,
    type: w.type,
    chain: (w.config as Record<string, unknown>)?.chain || 'unknown',
    linkedUser: w.linkedUser,
    createdAt: w.createdAt,
  }));
}

/**
 * Get wallet balance
 * For Base Sepolia, we query the USDC balance via RPC
 * Crossmint production API may support balance queries - this is a fallback
 */
export async function getWalletBalance(walletAddress: string): Promise<WalletBalance[]> {
  const chain = process.env.NEXT_PUBLIC_CROSSMINT_CHAIN || 'base-sepolia';

  try {
    // Base Sepolia RPC and USDC address
    const BASE_SEPOLIA_RPC = 'https://sepolia.base.org';
    // Testnet USDC on Base Sepolia (from faucet)
    const USDC_ADDRESS = '0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f';

    // balanceOf(address) selector + padded address
    const paddedAddress = walletAddress.slice(2).toLowerCase().padStart(64, '0');
    const data = `0x70a08231${paddedAddress}`;

    const response = await fetch(BASE_SEPOLIA_RPC, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_call',
        params: [{ to: USDC_ADDRESS, data }, 'latest'],
        id: 1,
      }),
    });

    const result = await response.json();

    if (result.result) {
      const balanceWei = BigInt(result.result);
      // USDC has 6 decimals
      const balanceUsdc = Number(balanceWei) / 1_000_000;

      return [{
        currency: 'usdc',
        amount: balanceUsdc.toFixed(2),
        chain,
      }];
    }

    return [];
  } catch (error) {
    logger.warn('Failed to fetch balance from Base Sepolia', { action: 'get_balance', error });
    return [];
  }
}

/**
 * Transfer USDC on Arc testnet
 * Uses direct RPC since Crossmint staging doesn't support Arc transfers
 */
export async function transferUsdc(
  _fromWallet: string,
  toAddress: string,
  amountUsdc: number,
  _chain: string = 'arc-testnet',
  proofHash?: string
): Promise<TransferResult> {
  // For Arc testnet, we use direct transfer via the demo wallet
  // In production, Crossmint would handle this after verifying the proof
  const ARC_RPC = 'https://rpc.testnet.arc.network';
  const USDC_ADDRESS = '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213';
  const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY;

  if (!DEMO_PRIVATE_KEY) {
    throw new Error('DEMO_WALLET_PRIVATE_KEY not configured for Arc transfers');
  }

  try {
    // Dynamic import viem for server-side
    const { createWalletClient, http, parseAbi } = await import('viem');
    const { privateKeyToAccount } = await import('viem/accounts');

    const ARC_TESTNET = {
      id: 5042002,
      name: 'Arc Testnet',
      nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
      rpcUrls: { default: { http: [ARC_RPC] } },
    };

    const account = privateKeyToAccount(DEMO_PRIVATE_KEY as `0x${string}`);

    const client = createWalletClient({
      account,
      chain: ARC_TESTNET,
      transport: http(),
    });

    // Convert USDC amount to wei (6 decimals)
    const amountWei = BigInt(Math.round(amountUsdc * 1_000_000));

    const hash = await client.writeContract({
      address: USDC_ADDRESS,
      abi: parseAbi(['function transfer(address to, uint256 amount) returns (bool)']),
      functionName: 'transfer',
      args: [toAddress as `0x${string}`, amountWei],
    });

    logger.info('Transfer executed', {
      action: 'transfer',
      txHash: hash,
      proofHash: proofHash || 'none',
      amount: amountUsdc,
      recipient: toAddress,
    });

    return {
      id: hash,
      status: 'success',
      txHash: hash,
      chain: 'arc-testnet',
      amount: amountUsdc.toString(),
      recipient: toAddress,
    };
  } catch (error) {
    logger.error('Arc transfer error', { action: 'transfer', error });
    throw new Error(`Arc transfer failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Transfer tokens via Crossmint API (for supported chains)
 *
 * This is the REAL Crossmint integration. It:
 * 1. Calls Crossmint's transfer API
 * 2. Crossmint handles gas, signing, and execution
 * 3. Returns transaction hash
 *
 * Supported chains: base-sepolia, polygon-amoy, base, polygon, arbitrum, optimism
 * Arc testnet requires Crossmint sales enablement.
 *
 * API: POST /wallets/{locator}/tokens/{token}/transfers
 */
export async function transferViaCrossmint(
  walletLocator: string,
  toAddress: string,
  amountUsdc: number,
  chain: string = 'base-sepolia',
  proofHash?: string
): Promise<TransferResult & { crossmintTransactionId?: string }> {
  if (!CROSSMINT_SERVER_KEY) {
    throw new Error('CROSSMINT_SERVER_KEY not configured');
  }

  // Crossmint uses chain:token format for token locator
  const tokenLocator = `${chain}:usdc`;

  logger.info('Initiating Crossmint transfer', {
    action: 'crossmint_transfer',
    wallet: walletLocator,
    to: toAddress,
    amount: amountUsdc,
    chain,
    tokenLocator,
    proofHash: proofHash || 'none',
  });

  try {
    // Step 1: Initiate transfer via Crossmint API (2025-06-09 version)
    const response = await fetch(
      `${CROSSMINT_API_URL}/${API_VERSION}/wallets/${walletLocator}/tokens/${tokenLocator}/transfers`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': CROSSMINT_SERVER_KEY,
        },
        body: JSON.stringify({
          recipient: toAddress,
          amount: amountUsdc.toString(),
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      logger.error('Crossmint transfer failed', {
        action: 'crossmint_transfer',
        status: response.status,
        error: errorText,
      });
      throw new Error(`Crossmint transfer failed: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    logger.info('Crossmint transfer initiated', {
      action: 'crossmint_transfer',
      transactionId: data.id,
      status: data.status,
    });

    // For API key auth (admin signer), transfer executes immediately
    // For delegated signers, would need approval step

    return {
      id: data.id,
      status: data.status === 'succeeded' ? 'success' : data.status,
      txHash: data.onChain?.txId,
      chain,
      amount: amountUsdc.toString(),
      recipient: toAddress,
      crossmintTransactionId: data.id,
    };
  } catch (error) {
    logger.error('Crossmint transfer error', { action: 'crossmint_transfer', error });
    throw error;
  }
}

/**
 * Poll for Crossmint transaction completion
 */
export async function waitForCrossmintTransaction(
  transactionId: string,
  maxAttempts: number = 30,
  intervalMs: number = 2000
): Promise<TransferResult> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const status = await getCrossmintTransactionStatus(transactionId);

    if (status.status === 'success' || status.status === 'failed') {
      return status;
    }

    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }

  throw new Error(`Transaction ${transactionId} did not complete within timeout`);
}

/**
 * Get Crossmint transaction status
 */
export async function getCrossmintTransactionStatus(
  transactionId: string
): Promise<TransferResult> {
  const response = await fetch(
    `${CROSSMINT_API_URL}/${API_VERSION}/transactions/${transactionId}`,
    {
      headers: {
        'X-API-KEY': CROSSMINT_SERVER_KEY!,
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to get transaction status: ${response.status}`);
  }

  const data = await response.json();

  return {
    id: data.id,
    status: data.status === 'succeeded' ? 'success' : data.status,
    txHash: data.onChain?.txId,
    chain: data.chain || 'unknown',
    amount: data.amount || '0',
    recipient: data.recipient || '',
  };
}

/**
 * Smart transfer function that chooses the right method based on chain support
 *
 * - For Crossmint-supported chains: Uses Crossmint API
 * - For unsupported chains (Arc): Uses direct on-chain transfer
 */
export async function smartTransfer(
  walletLocator: string,
  toAddress: string,
  amountUsdc: number,
  chain: string = 'arc-testnet',
  proofHash?: string
): Promise<TransferResult & { method: 'crossmint' | 'direct' }> {
  // Chains supported by Crossmint for transfers
  const crossmintSupportedChains = [
    'base-sepolia', 'polygon-amoy', 'base', 'polygon', 'arbitrum', 'optimism'
  ];

  if (crossmintSupportedChains.includes(chain) && INTEGRATION_MODE === 'crossmint') {
    logger.info('Using Crossmint-managed transfer', { action: 'smart_transfer', chain });
    const result = await transferViaCrossmint(walletLocator, toAddress, amountUsdc, chain, proofHash);
    return { ...result, method: 'crossmint' };
  } else {
    logger.info('Using direct on-chain transfer', { action: 'smart_transfer', chain });
    const result = await transferUsdc(walletLocator, toAddress, amountUsdc, chain, proofHash);
    return { ...result, method: 'direct' };
  }
}

/**
 * Get transfer status
 */
export async function getTransferStatus(transferId: string): Promise<TransferResult> {
  const response = await fetch(
    `${CROSSMINT_API_URL}/${API_VERSION}/transfers/${transferId}`,
    {
      headers: {
        'X-API-KEY': CROSSMINT_SERVER_KEY!,
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to get transfer status: ${response.status}`);
  }

  const data = await response.json();
  return {
    id: data.id,
    status: data.status,
    txHash: data.txHash || data.transactionHash,
    chain: data.chain,
    amount: data.amount,
    recipient: data.recipient,
  };
}
