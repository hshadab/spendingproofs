/**
 * Crossmint API Client
 *
 * Server-side client for Crossmint wallet and transfer operations.
 * Uses staging API with server-side API key.
 */

const CROSSMINT_API_URL = process.env.CROSSMINT_API_URL || 'https://staging.crossmint.com/api';
const CROSSMINT_SERVER_KEY = process.env.CROSSMINT_SERVER_KEY;

if (!CROSSMINT_SERVER_KEY) {
  console.warn('CROSSMINT_SERVER_KEY not set - Crossmint API calls will fail');
}

// API version for wallet endpoints (using latest)
const API_VERSION = '2022-06-09';

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
  const response = await fetch(`${CROSSMINT_API_URL}/${API_VERSION}/wallets`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-KEY': CROSSMINT_SERVER_KEY!,
    },
    body: JSON.stringify({
      type: 'evm-mpc-wallet',
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
    chain: 'evm', // MPC wallets are multi-chain
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
 * Fetches directly from Arc testnet since Crossmint staging API doesn't support MPC wallet balances
 */
export async function getWalletBalance(walletAddress: string): Promise<WalletBalance[]> {
  try {
    // Query Arc testnet USDC balance directly via RPC
    const ARC_RPC = 'https://rpc.testnet.arc.network';
    const USDC_ADDRESS = '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213';

    // balanceOf(address) selector + padded address
    const paddedAddress = walletAddress.slice(2).toLowerCase().padStart(64, '0');
    const data = `0x70a08231${paddedAddress}`;

    const response = await fetch(ARC_RPC, {
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
        chain: 'arc-testnet',
      }];
    }

    return [];
  } catch (error) {
    console.warn('Failed to fetch balance from Arc:', error);
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

    console.log(`Transfer executed: ${hash}, proofHash: ${proofHash || 'none'}`);

    return {
      id: hash,
      status: 'success',
      txHash: hash,
      chain: 'arc-testnet',
      amount: amountUsdc.toString(),
      recipient: toAddress,
    };
  } catch (error) {
    console.error('Arc transfer error:', error);
    throw new Error(`Arc transfer failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
