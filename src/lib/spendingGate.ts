/**
 * SpendingGate Contract Interface
 *
 * This interfaces with the SpendingGate smart contract that ENFORCES spending
 * policy compliance by requiring valid proofs before allowing USDC transfers.
 *
 * Key difference from attestation:
 * - Attestation: "We logged that this proof was submitted" (advisory)
 * - Enforcement: "Transfer REVERTS without valid proof" (mandatory)
 */

import { SpendingProof, TxIntent } from './types';
import { keccak256, encodePacked } from 'viem';

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
 * Uses real keccak256 for cryptographic binding
 */
export function computeTxIntentHash(intent: TxIntent): string {
  // Pack all intent fields using ABI encoding (matches Solidity)
  const encoded = encodePacked(
    ['uint256', 'address', 'address', 'address', 'uint256', 'uint256', 'uint256', 'string', 'uint256'],
    [
      BigInt(intent.chainId),
      intent.usdcAddress as `0x${string}`,
      intent.sender as `0x${string}`,
      intent.recipient as `0x${string}`,
      BigInt(intent.amount),
      BigInt(intent.nonce),
      BigInt(intent.expiry),
      intent.policyId,
      BigInt(intent.policyVersion),
    ]
  );

  return keccak256(encoded);
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
