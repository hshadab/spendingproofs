/**
 * Centralized Configuration
 *
 * All environment variables and hardcoded values consolidated in one place.
 * This prevents configuration sprawl and makes it easier to manage deployments.
 */

type Address = `0x${string}`;

/**
 * Arc Testnet Chain Configuration
 */
export const ARC_CHAIN = {
  id: 5042002,
  name: 'Arc Testnet',
  nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
  rpcUrl: process.env.NEXT_PUBLIC_ARC_RPC || 'https://rpc.testnet.arc.network',
  explorerUrl: 'https://testnet.arcscan.app',
} as const;

/**
 * Contract Addresses on Arc Testnet
 */
export const ADDRESSES = {
  /** USDC token contract (6 decimals) */
  usdc: (process.env.NEXT_PUBLIC_USDC_ADDRESS ||
    '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213') as Address,

  /** Proof attestation contract - logs proof submissions */
  proofAttestation: (process.env.NEXT_PUBLIC_PROOF_ATTESTATION ||
    '0xBE9a5DF7C551324CB872584C6E5bF56799787952') as Address,

  /** Spending gate wallet - gated USDC transfers */
  spendingGate: (process.env.NEXT_PUBLIC_SPENDING_GATE_ADDRESS ||
    '0x6A47D13593c00359a1c5Fc6f9716926aF184d138') as Address,

  /** Arc agent contract (demo/legacy) */
  arcAgent: (process.env.NEXT_PUBLIC_ARC_AGENT ||
    '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384') as Address,

  /** Demo merchant address for receiving payments */
  demoMerchant: (process.env.NEXT_PUBLIC_DEMO_MERCHANT ||
    '0x8ba1f109551bD432803012645Ac136ddd64DBA72') as Address,
} as const;

/**
 * API Configuration
 */
export const API_CONFIG = {
  /** Prover backend URL */
  proverBackendUrl: process.env.PROVER_BACKEND_URL || 'http://localhost:3001',

  /** Public prover URL (client-side) */
  publicProverUrl: process.env.NEXT_PUBLIC_PROVER_URL,

  /** Jolt Atlas prover URL */
  joltAtlasUrl:
    process.env.NEXT_PUBLIC_JOLT_ATLAS_URL ||
    'https://spendingproofs-prover.onrender.com',

  /** Crossmint API URL (use production for sk_production_ keys) */
  crossmintApiUrl:
    process.env.CROSSMINT_API_URL || 'https://www.crossmint.com/api',

  /** Crossmint API version */
  crossmintApiVersion: '2025-06-09',
} as const;

/**
 * Authentication Configuration
 */
export const AUTH_CONFIG = {
  /** Whether signature authentication is required */
  requireSignatureAuth: process.env.REQUIRE_SIGNATURE_AUTH === 'true',

  /** Signature expiry time in milliseconds (5 minutes) */
  signatureExpiryMs: 5 * 60 * 1000,

  /** Clock skew tolerance in milliseconds (10 seconds) */
  clockSkewToleranceMs: 10 * 1000,

  /** Allowed prover addresses (comma-separated in env) */
  allowedAddresses: (process.env.ALLOWED_PROVER_ADDRESSES || '')
    .split(',')
    .map((addr) => addr.trim().toLowerCase())
    .filter(Boolean),
} as const;

/**
 * Cache Configuration
 */
export const CACHE_CONFIG = {
  /** Maximum number of cached proofs */
  maxSize: 100,

  /** Proof cache TTL in milliseconds (5 minutes - reduced from 15 for better freshness) */
  ttlMs: 5 * 60 * 1000,

  /** Whether caching is enabled */
  enabled: process.env.PROOF_CACHE_ENABLED !== 'false',
} as const;

/**
 * Retry Configuration
 */
export const RETRY_CONFIG = {
  /** Maximum retry attempts */
  maxAttempts: 3,

  /** Initial delay in milliseconds */
  initialDelayMs: 1000,

  /** Maximum delay in milliseconds */
  maxDelayMs: 30000,

  /** Backoff multiplier */
  backoffMultiplier: 2,

  /** Jitter factor (0-1) */
  jitter: 0.1,
} as const;

/**
 * Prover-specific retry configuration
 */
export const PROVER_RETRY_CONFIG = {
  maxAttempts: 3,
  initialDelayMs: 2000,
  maxDelayMs: 10000,
  backoffMultiplier: 1.5,
  jitter: 0.2,
} as const;

/**
 * Environment checks
 */
export const ENV = {
  /** Whether we're in production */
  isProduction: process.env.NODE_ENV === 'production',

  /** Whether we're in development */
  isDevelopment: process.env.NODE_ENV === 'development',

  /** Whether we're in test mode */
  isTest: process.env.NODE_ENV === 'test',

  /** Whether logging is enabled */
  loggingEnabled: process.env.NODE_ENV !== 'test',
} as const;

/**
 * Get explorer URL for a transaction
 */
export function getExplorerTxUrl(txHash: string): string {
  return `${ARC_CHAIN.explorerUrl}/tx/${txHash}`;
}

/**
 * Get explorer URL for an address
 */
export function getExplorerAddressUrl(address: string): string {
  return `${ARC_CHAIN.explorerUrl}/address/${address}`;
}

/**
 * Validate required environment variables
 * Call this on app startup to fail fast
 */
export function validateEnv(): { valid: boolean; missing: string[] } {
  const required: string[] = [];
  const missing: string[] = [];

  // Only require these in production
  if (ENV.isProduction) {
    required.push('CROSSMINT_SERVER_KEY');
  }

  for (const key of required) {
    if (!process.env[key]) {
      missing.push(key);
    }
  }

  return {
    valid: missing.length === 0,
    missing,
  };
}
