/**
 * Cryptographic Utilities
 *
 * Secure random generation and related crypto helpers.
 * Uses crypto.getRandomValues() for cryptographically secure randomness.
 */

/**
 * Generate a cryptographically secure random hex string
 * @param byteLength - Number of random bytes (output will be 2x this in hex chars)
 * @returns Hex string without 0x prefix
 */
export function generateSecureHex(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Generate a cryptographically secure bytes32 hash (for mock proofs, IDs, etc.)
 * @returns 0x-prefixed 64-character hex string (32 bytes)
 */
export function generateSecureBytes32(): `0x${string}` {
  return `0x${generateSecureHex(32)}`;
}

/**
 * Generate a cryptographically secure random ID
 * @param prefix - Optional prefix for the ID
 * @returns Random ID string
 */
export function generateSecureId(prefix?: string): string {
  const randomPart = generateSecureHex(8);
  return prefix ? `${prefix}_${randomPart}` : randomPart;
}

/**
 * Generate a mock proof hash for demo/testing purposes
 * WARNING: Mock proofs are NOT cryptographically valid and cannot be verified
 * @returns 0x-prefixed 64-character hex string
 */
export function generateMockProofHash(): `0x${string}` {
  return generateSecureBytes32();
}

/**
 * Generate mock input/output hashes for demo purposes
 * @returns Object with inputHash and outputHash
 */
export function generateMockHashes(): {
  inputHash: `0x${string}`;
  outputHash: `0x${string}`;
} {
  return {
    inputHash: generateSecureBytes32(),
    outputHash: generateSecureBytes32(),
  };
}

/**
 * Generate cryptographically secure jitter value for retry logic
 * @param range - Maximum jitter range (positive or negative)
 * @returns Random value between -range and +range
 */
export function generateSecureJitter(range: number): number {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  // Convert to number between 0 and 1
  const randomValue =
    (bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)) >>> 0;
  const normalized = randomValue / 0xffffffff;
  // Scale to -range to +range
  return (normalized * 2 - 1) * range;
}

/**
 * Check if we're in an environment that supports crypto.getRandomValues
 */
export function isCryptoAvailable(): boolean {
  return (
    typeof crypto !== 'undefined' &&
    typeof crypto.getRandomValues === 'function'
  );
}
