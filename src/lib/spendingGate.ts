/**
 * Mock SpendingGate Contract Interface
 *
 * This simulates a smart contract that ENFORCES spending policy compliance
 * by requiring valid proofs before allowing USDC transfers.
 *
 * Key difference from attestation:
 * - Attestation: "We logged that this proof was submitted" (advisory)
 * - Enforcement: "Transfer REVERTS without valid proof" (mandatory)
 */

import { SpendingProof, TxIntent } from './types';

export type { TxIntent };

export interface GatedTransferResult {
  success: boolean;
  txHash?: string;
  revertReason?: string;
  gasUsed?: number;
}

export interface SpendingGateState {
  usedNonces: Set<string>;
  registeredPolicies: Map<string, { modelHash: string; vkHash: string; version: number }>;
}

// Mock contract state
const contractState: SpendingGateState = {
  usedNonces: new Set(),
  registeredPolicies: new Map([
    ['default-spending-policy', {
      modelHash: '0x7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b',
      vkHash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
      version: 1,
    }],
  ]),
};

/**
 * Compute txIntentHash - binds proof to specific transaction intent
 */
export function computeTxIntentHash(intent: TxIntent): string {
  const packed = [
    intent.chainId.toString(16).padStart(8, '0'),
    intent.usdcAddress.slice(2).toLowerCase(),
    intent.sender.slice(2).toLowerCase(),
    intent.recipient.slice(2).toLowerCase(),
    intent.amount.toString(16).padStart(64, '0'),
    intent.nonce.toString(16).padStart(16, '0'),
    intent.expiry.toString(16).padStart(8, '0'),
    intent.policyId,
    intent.policyVersion.toString(16).padStart(4, '0'),
  ].join('');

  // Simple hash simulation (in real contract, use keccak256)
  let hash = 0;
  for (let i = 0; i < packed.length; i++) {
    const char = packed.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return '0x' + Math.abs(hash).toString(16).padStart(64, '0');
}

/**
 * Mock SpendingGate.gatedTransfer
 *
 * Simulates the on-chain enforcement logic:
 * 1. Check proof is provided
 * 2. Verify txIntentHash matches
 * 3. Check nonce not already used
 * 4. Check expiry not passed
 * 5. Verify policy is registered and version matches
 * 6. Execute transfer
 */
export async function gatedTransfer(
  intent: TxIntent,
  proof: SpendingProof | null,
  options: { skipProof?: boolean; modifyAmount?: boolean; replayNonce?: boolean } = {}
): Promise<GatedTransferResult> {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 500));

  // 1. Check proof is provided
  if (!proof || options.skipProof) {
    return {
      success: false,
      revertReason: 'SpendingGate: PROOF_REQUIRED - Transfer reverted without valid spending proof',
    };
  }

  // 2. Check proof decision
  if (!proof.decision?.shouldBuy) {
    return {
      success: false,
      revertReason: 'SpendingGate: POLICY_REJECTED - Spending policy did not approve this transaction',
    };
  }

  // 3. Compute and verify txIntentHash
  const computedHash = computeTxIntentHash(intent);

  // If modifyAmount option is set, simulate amount manipulation
  if (options.modifyAmount) {
    return {
      success: false,
      revertReason: `SpendingGate: INTENT_MISMATCH - txIntentHash mismatch. Proof was generated for different amount.`,
    };
  }

  // 4. Check nonce not already used (replay protection)
  const nonceKey = `${intent.sender}-${intent.nonce}`;
  if (contractState.usedNonces.has(nonceKey) || options.replayNonce) {
    return {
      success: false,
      revertReason: 'SpendingGate: NONCE_ALREADY_USED - This proof has already been consumed',
    };
  }

  // 5. Check expiry
  const now = Math.floor(Date.now() / 1000);
  if (intent.expiry < now) {
    return {
      success: false,
      revertReason: 'SpendingGate: PROOF_EXPIRED - Transaction intent has expired',
    };
  }

  // 6. Check policy is registered
  const policy = contractState.registeredPolicies.get(intent.policyId);
  if (!policy) {
    return {
      success: false,
      revertReason: 'SpendingGate: UNKNOWN_POLICY - Policy ID not registered in PolicyRegistry',
    };
  }

  // 7. Check policy version
  if (policy.version !== intent.policyVersion) {
    return {
      success: false,
      revertReason: `SpendingGate: VERSION_MISMATCH - Expected policy v${policy.version}, got v${intent.policyVersion}`,
    };
  }

  // 8. Mark nonce as used
  contractState.usedNonces.add(nonceKey);

  // 9. Execute transfer (mock)
  const mockTxHash = '0x' + Array.from({ length: 64 }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join('');

  return {
    success: true,
    txHash: mockTxHash,
    gasUsed: 85000 + Math.floor(Math.random() * 10000),
  };
}

/**
 * Reset mock contract state (for demo purposes)
 */
export function resetContractState(): void {
  contractState.usedNonces.clear();
}

/**
 * Register a policy in the mock registry
 */
export function registerPolicy(
  policyId: string,
  modelHash: string,
  vkHash: string,
  version: number
): void {
  contractState.registeredPolicies.set(policyId, { modelHash, vkHash, version });
}

/**
 * Check if a policy is registered
 */
export function isPolicyRegistered(policyId: string): boolean {
  return contractState.registeredPolicies.has(policyId);
}
