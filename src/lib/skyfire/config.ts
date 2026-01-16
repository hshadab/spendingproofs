/**
 * Skyfire Configuration
 *
 * Configuration for Skyfire KYA (Know Your Agent) and payment integration.
 * Supports both sandbox and production environments.
 */

import type { SkyfireConfig } from './types';

/**
 * Skyfire API configuration
 */
export const SKYFIRE_CONFIG: SkyfireConfig = {
  apiUrl: process.env.SKYFIRE_API_URL || 'https://api.skyfire.xyz',
  mcpUrl: process.env.SKYFIRE_MCP_URL || 'https://mcp.skyfire.xyz',
  environment: (process.env.SKYFIRE_ENVIRONMENT || 'sandbox') as 'sandbox' | 'production',
  enabled: process.env.NEXT_PUBLIC_SKYFIRE_ENABLED === 'true',
};

/**
 * Check if Skyfire integration is properly configured
 */
export function isSkyfireConfigured(): boolean {
  return !!(
    process.env.SKYFIRE_API_KEY &&
    SKYFIRE_CONFIG.enabled
  );
}

/**
 * Skyfire API headers
 * Uses skyfire-api-key header for authentication
 */
export function getSkyfireHeaders(): HeadersInit {
  const apiKey = process.env.SKYFIRE_API_KEY;
  if (!apiKey) {
    throw new Error('SKYFIRE_API_KEY not configured');
  }

  return {
    'Content-Type': 'application/json',
    'skyfire-api-key': apiKey,
  };
}

/**
 * Skyfire Official Seller Service (Sandbox)
 * Used for demo token creation
 */
export const SKYFIRE_SELLER_SERVICE_ID = '3b622b2f-7a2d-4ee5-86c5-58d3b8bdf73d';

/**
 * Demo mode configuration
 * Used when Skyfire API is not available or for testing
 */
export const DEMO_CONFIG = {
  // Demo agent configuration
  demoAgentId: 'demo-agent-zkml-001',
  demoAgentName: 'zkML Demo Agent',
  demoWalletAddress: process.env.NEXT_PUBLIC_SKYFIRE_DEMO_WALLET_ADDRESS || '0x0e9AFe2499211c3E35e570968d1047Fcf7488c60',

  // Demo timing (milliseconds)
  kyaVerificationDelay: 1500,
  payTokenGenerationDelay: 1000,
  transferExecutionDelay: 2000,

  // Demo recipient for transfers
  demoRecipient: '0x742d35Cc6634C0532925a3b844Bc9e7595f1E5b8',
  demoAmount: 10.00, // $10 USDC
};

/**
 * Skyfire capabilities for demo agent
 */
export const DEMO_CAPABILITIES = [
  'payment:transfer',
  'payment:receive',
  'api:purchase',
  'service:consume',
];

/**
 * Get Skyfire explorer URL for a transaction
 */
export function getSkyfireExplorerUrl(transactionId: string): string {
  const baseUrl = SKYFIRE_CONFIG.environment === 'production'
    ? 'https://explorer.skyfire.xyz'
    : 'https://explorer.sandbox.skyfire.xyz';
  return `${baseUrl}/tx/${transactionId}`;
}
