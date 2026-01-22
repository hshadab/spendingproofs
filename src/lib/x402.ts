/**
 * x402 Payment Protocol Integration
 *
 * Implements x402 (HTTP 402 Payment Required) for USDC payments on Base Sepolia.
 * Used for AI agent autonomous payments with zkML spending proof verification.
 *
 * @see https://x402.org
 * @see https://docs.cdp.coinbase.com/x402/welcome
 */

import { createPublicClient, createWalletClient, http, parseUnits, formatUnits } from 'viem';
import { baseSepolia } from 'viem/chains';
import { privateKeyToAccount } from 'viem/accounts';
import type { Hex, Address } from 'viem';

// Base Sepolia USDC contract address
export const BASE_SEPOLIA_USDC = '0x036CbD53842c5426634e7929541eC2318f3dCF7e' as const;

// Coinbase facilitator URL for fee-free USDC payments
export const COINBASE_FACILITATOR_URL = 'https://x402.coinbase.com' as const;

// Base Sepolia chain configuration
export const BASE_SEPOLIA_CONFIG = {
  chainId: baseSepolia.id,
  name: 'Base Sepolia',
  network: 'base-sepolia',
  rpcUrl: 'https://sepolia.base.org',
  explorerUrl: 'https://sepolia.basescan.org',
  usdc: BASE_SEPOLIA_USDC,
};

/**
 * x402 Payment Required response structure
 */
export interface X402PaymentRequired {
  version: string;
  network: string;
  payTo: Address;
  maxAmountRequired: string;
  asset: Address;
  description?: string;
  resource?: string;
  scheme: string;
  extra?: Record<string, unknown>;
}

/**
 * x402 Payment payload structure (ERC-3009 TransferWithAuthorization)
 */
export interface X402PaymentPayload {
  signature: Hex;
  payload: {
    from: Address;
    to: Address;
    value: bigint;
    validAfter: bigint;
    validBefore: bigint;
    nonce: Hex;
  };
}

/**
 * x402 Transaction result
 */
export interface X402TransactionResult {
  success: boolean;
  txHash?: string;
  settlementTxHash?: string;
  error?: string;
  paymentRequired?: X402PaymentRequired;
  amountPaid?: string;
  recipient?: string;
  explorerUrl?: string;
}

/**
 * Create a public client for Base Sepolia
 */
export function createBaseSepoliaPublicClient() {
  return createPublicClient({
    chain: baseSepolia,
    transport: http(BASE_SEPOLIA_CONFIG.rpcUrl),
  });
}

/**
 * Create a wallet client for Base Sepolia
 */
export function createBaseSepoliaWalletClient(privateKey: Hex) {
  const account = privateKeyToAccount(privateKey);
  return createWalletClient({
    account,
    chain: baseSepolia,
    transport: http(BASE_SEPOLIA_CONFIG.rpcUrl),
  });
}

/**
 * Get USDC balance on Base Sepolia
 */
export async function getBaseSepoliaUsdcBalance(address: Address): Promise<string> {
  const publicClient = createBaseSepoliaPublicClient();

  const balance = await publicClient.readContract({
    address: BASE_SEPOLIA_USDC,
    abi: [
      {
        name: 'balanceOf',
        type: 'function',
        stateMutability: 'view',
        inputs: [{ name: 'account', type: 'address' }],
        outputs: [{ name: '', type: 'uint256' }],
      },
    ],
    functionName: 'balanceOf',
    args: [address],
  });

  // USDC has 6 decimals
  return formatUnits(balance as bigint, 6);
}

/**
 * Parse x402 Payment-Required header
 */
export function parsePaymentRequiredHeader(header: string): X402PaymentRequired | null {
  try {
    // Header is base64 encoded JSON
    const decoded = Buffer.from(header, 'base64').toString('utf-8');
    return JSON.parse(decoded);
  } catch {
    return null;
  }
}

/**
 * Create x402 Payment-Signature header
 */
export function createPaymentSignatureHeader(payload: X402PaymentPayload): string {
  const json = JSON.stringify({
    signature: payload.signature,
    payload: {
      from: payload.payload.from,
      to: payload.payload.to,
      value: payload.payload.value.toString(),
      validAfter: payload.payload.validAfter.toString(),
      validBefore: payload.payload.validBefore.toString(),
      nonce: payload.payload.nonce,
    },
  });
  return Buffer.from(json).toString('base64');
}

/**
 * ERC-3009 TransferWithAuthorization type data for signing
 */
const TRANSFER_WITH_AUTHORIZATION_TYPEHASH = {
  name: 'TransferWithAuthorization',
  fields: [
    { name: 'from', type: 'address' },
    { name: 'to', type: 'address' },
    { name: 'value', type: 'uint256' },
    { name: 'validAfter', type: 'uint256' },
    { name: 'validBefore', type: 'uint256' },
    { name: 'nonce', type: 'bytes32' },
  ],
} as const;

/**
 * Get USDC domain for EIP-712 signing on Base Sepolia
 */
export function getUsdcDomain() {
  return {
    name: 'USD Coin',
    version: '2',
    chainId: baseSepolia.id,
    verifyingContract: BASE_SEPOLIA_USDC,
  };
}

/**
 * Sign an ERC-3009 TransferWithAuthorization for x402 payment
 */
export async function signX402Payment(
  privateKey: Hex,
  to: Address,
  amount: string, // Amount in USDC (e.g., "0.85")
  validForSeconds: number = 300 // 5 minutes default
): Promise<X402PaymentPayload> {
  const account = privateKeyToAccount(privateKey);
  const walletClient = createBaseSepoliaWalletClient(privateKey);

  const value = parseUnits(amount, 6); // USDC has 6 decimals
  const now = BigInt(Math.floor(Date.now() / 1000));
  const validAfter = now - BigInt(60); // Valid from 1 minute ago
  const validBefore = now + BigInt(validForSeconds);

  // Generate random nonce
  const nonceBytes = new Uint8Array(32);
  crypto.getRandomValues(nonceBytes);
  const nonce = ('0x' + Array.from(nonceBytes).map(b => b.toString(16).padStart(2, '0')).join('')) as Hex;

  const domain = getUsdcDomain();

  const signature = await walletClient.signTypedData({
    account,
    domain,
    types: {
      TransferWithAuthorization: [
        { name: 'from', type: 'address' },
        { name: 'to', type: 'address' },
        { name: 'value', type: 'uint256' },
        { name: 'validAfter', type: 'uint256' },
        { name: 'validBefore', type: 'uint256' },
        { name: 'nonce', type: 'bytes32' },
      ],
    },
    primaryType: 'TransferWithAuthorization',
    message: {
      from: account.address,
      to,
      value,
      validAfter,
      validBefore,
      nonce,
    },
  });

  return {
    signature,
    payload: {
      from: account.address,
      to,
      value,
      validAfter,
      validBefore,
      nonce,
    },
  };
}

/**
 * Execute a direct USDC transfer on Base Sepolia (fallback if x402 facilitator unavailable)
 */
export async function executeDirectUsdcTransfer(
  privateKey: Hex,
  to: Address,
  amount: string // Amount in USDC
): Promise<X402TransactionResult> {
  try {
    const account = privateKeyToAccount(privateKey);
    const walletClient = createBaseSepoliaWalletClient(privateKey);
    const publicClient = createBaseSepoliaPublicClient();

    const value = parseUnits(amount, 6);

    // ERC-20 transfer
    const hash = await walletClient.writeContract({
      address: BASE_SEPOLIA_USDC,
      abi: [
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
      ],
      functionName: 'transfer',
      args: [to, value],
    });

    // Wait for confirmation
    const receipt = await publicClient.waitForTransactionReceipt({ hash });

    return {
      success: receipt.status === 'success',
      txHash: hash,
      amountPaid: amount,
      recipient: to,
      explorerUrl: `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${hash}`,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Transfer failed',
    };
  }
}

/**
 * Make an x402 payment request
 *
 * This handles the full x402 flow:
 * 1. Make initial request
 * 2. If 402, parse payment requirements
 * 3. Sign payment authorization
 * 4. Retry with payment header
 */
export async function makeX402Request(
  url: string,
  privateKey: Hex,
  options: RequestInit = {}
): Promise<{
  response: Response;
  paymentMade: boolean;
  paymentDetails?: {
    amount: string;
    recipient: string;
    txHash?: string;
  };
}> {
  // First request - may get 402
  const initialResponse = await fetch(url, options);

  if (initialResponse.status !== 402) {
    return { response: initialResponse, paymentMade: false };
  }

  // Parse 402 payment requirements
  const paymentRequiredHeader = initialResponse.headers.get('X-Payment-Required')
    || initialResponse.headers.get('Payment-Required');

  if (!paymentRequiredHeader) {
    throw new Error('402 response missing Payment-Required header');
  }

  const paymentRequired = parsePaymentRequiredHeader(paymentRequiredHeader);
  if (!paymentRequired) {
    throw new Error('Could not parse Payment-Required header');
  }

  // Sign payment
  const amountUsdc = formatUnits(BigInt(paymentRequired.maxAmountRequired), 6);
  const paymentPayload = await signX402Payment(
    privateKey,
    paymentRequired.payTo,
    amountUsdc
  );

  // Retry with payment
  const paymentHeader = createPaymentSignatureHeader(paymentPayload);
  const retryResponse = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'X-Payment': paymentHeader,
    },
  });

  return {
    response: retryResponse,
    paymentMade: true,
    paymentDetails: {
      amount: amountUsdc,
      recipient: paymentRequired.payTo,
    },
  };
}

/**
 * Get explorer URL for a transaction on Base Sepolia
 */
export function getBaseSepoliaExplorerUrl(txHash: string): string {
  return `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${txHash}`;
}

/**
 * Get explorer URL for an address on Base Sepolia
 */
export function getBaseSepoliaAddressUrl(address: string): string {
  return `${BASE_SEPOLIA_CONFIG.explorerUrl}/address/${address}`;
}
