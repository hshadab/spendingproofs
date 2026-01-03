import { verifyMessage, keccak256, toBytes } from 'viem';
import type { SignedProveRequest, ProveErrorCode } from './types';

const SIGNATURE_EXPIRY_MS = 5 * 60 * 1000; // 5 minutes

// Optional allowlist - if empty, all addresses are allowed
const ALLOWED_ADDRESSES: Set<string> = new Set(
  (typeof process !== 'undefined' && process.env?.ALLOWED_PROVER_ADDRESSES || '')
    .split(',')
    .map((addr) => addr.trim().toLowerCase())
    .filter(Boolean)
);

export interface SignatureVerificationResult {
  valid: boolean;
  error?: string;
  code?: ProveErrorCode;
  address?: `0x${string}`;
}

/**
 * Create the message to be signed for proof requests
 */
export function createProveMessage(inputs: number[], tag: string, timestamp: number): string {
  const inputHash = keccak256(toBytes(JSON.stringify(inputs)));
  return `Spending Proofs Authentication\n\nAction: Generate proof\nTag: ${tag}\nInput Hash: ${inputHash}\nTimestamp: ${timestamp}`;
}

/**
 * Verify a signed proof request
 */
export async function verifyProveRequest(
  request: SignedProveRequest
): Promise<SignatureVerificationResult> {
  const { inputs, tag, address, timestamp, signature } = request;

  // Check timestamp is not expired
  const now = Date.now();
  if (now - timestamp > SIGNATURE_EXPIRY_MS) {
    return {
      valid: false,
      error: `Signature expired. Request was signed ${Math.round((now - timestamp) / 1000)}s ago, max allowed is ${SIGNATURE_EXPIRY_MS / 1000}s`,
      code: 'SIGNATURE_EXPIRED',
    };
  }

  // Check timestamp is not in the future (clock skew tolerance: 30s)
  if (timestamp > now + 30000) {
    return {
      valid: false,
      error: 'Signature timestamp is in the future',
      code: 'INVALID_SIGNATURE',
    };
  }

  // Reconstruct the expected message
  const expectedMessage = createProveMessage(inputs, tag, timestamp);

  // Verify the signature
  try {
    const isValid = await verifyMessage({
      address,
      message: expectedMessage,
      signature,
    });

    if (!isValid) {
      return {
        valid: false,
        error: 'Invalid signature',
        code: 'INVALID_SIGNATURE',
      };
    }
  } catch (err) {
    return {
      valid: false,
      error: `Signature verification failed: ${err instanceof Error ? err.message : 'Unknown error'}`,
      code: 'INVALID_SIGNATURE',
    };
  }

  // Check allowlist (if configured)
  if (ALLOWED_ADDRESSES.size > 0 && !ALLOWED_ADDRESSES.has(address.toLowerCase())) {
    return {
      valid: false,
      error: 'Address not in allowlist',
      code: 'ADDRESS_NOT_ALLOWED',
    };
  }

  return {
    valid: true,
    address,
  };
}

/**
 * Check if signature auth is enabled
 */
export function isSignatureAuthEnabled(): boolean {
  return process.env.REQUIRE_SIGNATURE_AUTH === 'true';
}
