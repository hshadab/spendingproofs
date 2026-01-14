/**
 * Agent Commerce Kit Client
 *
 * Initializes and configures the ACK SDK for use with Arc Testnet
 */

import { ARC_CHAIN } from '../config';

/**
 * ACK Configuration for Arc Testnet
 */
export const ACK_CONFIG = {
  /** Network configuration */
  network: {
    chainId: ARC_CHAIN.id,
    rpcUrl: ARC_CHAIN.rpcUrl,
    name: ARC_CHAIN.name,
  },
  /** Identity configuration */
  identity: {
    /** DID method to use */
    didMethod: 'key' as const,
    /** Credential issuer (self-issued for demo) */
    issuer: 'self',
  },
  /** Receipt configuration */
  receipts: {
    /** Receipt issuer (self-issued for demo) */
    issuer: 'self',
    /** Receipt validity period (24 hours) */
    validityPeriodMs: 24 * 60 * 60 * 1000,
  },
} as const;

/**
 * Generate a deterministic DID from a wallet address
 * Uses did:key method derived from the address
 */
export function generateAgentDid(walletAddress: string): string {
  // For demo purposes, create a did:key from the wallet address
  // In production, this would use proper key derivation
  const addressHash = walletAddress.toLowerCase().slice(2);
  return `did:key:z${addressHash}`;
}

/**
 * Format a DID for display (truncated)
 */
export function formatDid(did: string, maxLength = 24): string {
  if (did.length <= maxLength) return did;
  const prefix = did.slice(0, 12);
  const suffix = did.slice(-8);
  return `${prefix}...${suffix}`;
}

/**
 * Get the current timestamp in ISO format
 */
export function getIsoTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Get expiration timestamp (24 hours from now)
 */
export function getExpirationTimestamp(): string {
  const expiry = new Date(Date.now() + ACK_CONFIG.receipts.validityPeriodMs);
  return expiry.toISOString();
}
