/**
 * Circle Programmable Wallets SDK Configuration
 *
 * Server-side and client-side utilities for Circle wallet integration.
 */

// Circle API configuration
export const CIRCLE_CONFIG = {
  apiUrl: 'https://api.circle.com/v1/w3s',
  testApiUrl: 'https://api.circle.com/v1/w3s', // Same for test environment
} as const;

// Wallet types
export interface CircleWallet {
  id: string;
  address: string;
  blockchain: string;
  state: 'LIVE' | 'FROZEN';
  createDate: string;
  updateDate: string;
  custodyType: 'DEVELOPER' | 'ENDUSER';
  accountType: 'SCA' | 'EOA';
}

export interface CircleWalletSet {
  id: string;
  custodyType: string;
  name: string;
  createDate: string;
  updateDate: string;
}

export interface CircleTransaction {
  id: string;
  state: 'INITIATED' | 'PENDING_RISK_SCREENING' | 'CONFIRMED' | 'COMPLETE' | 'FAILED' | 'CANCELLED';
  txHash?: string;
  blockchain: string;
  createDate: string;
}

export interface CreateWalletParams {
  walletSetId: string;
  blockchain: string;
  accountType?: 'SCA' | 'EOA';
  metadata?: Array<{ name: string; refId: string }>;
}

export interface TransferParams {
  walletId: string;
  tokenAddress: string;
  destinationAddress: string;
  amount: string;
  feeLevel?: 'LOW' | 'MEDIUM' | 'HIGH';
}

// Error types
export class CircleAPIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code?: string
  ) {
    super(message);
    this.name = 'CircleAPIError';
  }
}

/**
 * Parse Circle API key format
 * Format: TEST_API_KEY:entitySecret:ciphertext or API_KEY:entitySecret:ciphertext
 */
export function parseCircleApiKey(apiKey: string): {
  apiKey: string;
  entitySecret: string;
  isTest: boolean;
} {
  const parts = apiKey.split(':');
  if (parts.length < 3) {
    throw new Error('Invalid Circle API key format');
  }

  const isTest = parts[0] === 'TEST_API_KEY';
  // The full API key includes the prefix
  const fullApiKey = `${parts[0]}:${parts[1]}:${parts[2]}`;
  const entitySecret = parts[1];

  return {
    apiKey: fullApiKey,
    entitySecret,
    isTest,
  };
}

/**
 * Get the appropriate Circle API URL based on environment
 */
export function getCircleApiUrl(isTest: boolean): string {
  return isTest ? CIRCLE_CONFIG.testApiUrl : CIRCLE_CONFIG.apiUrl;
}

/**
 * Format USDC amount for Circle API (6 decimals)
 */
export function formatUsdcAmount(amount: number): string {
  return amount.toFixed(6);
}

/**
 * Parse USDC amount from Circle API response
 */
export function parseUsdcAmount(amount: string): number {
  return parseFloat(amount);
}

/**
 * Circle blockchain identifiers
 */
export const CIRCLE_BLOCKCHAINS = {
  ETHEREUM: 'ETH',
  ETHEREUM_SEPOLIA: 'ETH-SEPOLIA',
  POLYGON: 'MATIC',
  POLYGON_AMOY: 'MATIC-AMOY',
  AVALANCHE: 'AVAX',
  AVALANCHE_FUJI: 'AVAX-FUJI',
  // Note: Arc may need custom configuration with Circle
} as const;

/**
 * Get USDC contract address for a blockchain
 */
export function getUsdcAddress(blockchain: string): string {
  const addresses: Record<string, string> = {
    'ETH': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'ETH-SEPOLIA': '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
    'MATIC': '0x3c499c542cef5e3811e1192ce70d8cc03d5c3359',
    'MATIC-AMOY': '0x41e94eb019c0762f9bfcf9fb1e58725bfb0e7582',
  };
  return addresses[blockchain] || '';
}
