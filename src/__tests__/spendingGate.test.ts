import { describe, it, expect, beforeEach } from 'vitest';
import {
  gatedTransfer,
  computeTxIntentHash,
  resetContractState,
  registerPolicy,
  isPolicyRegistered,
  TxIntent,
} from '../lib/spendingGate';
import { SpendingProof } from '../lib/types';

describe('SpendingGate', () => {
  const validIntent: TxIntent = {
    chainId: 5042002,
    usdcAddress: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    sender: '0x1234567890123456789012345678901234567890',
    recipient: '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
    amount: BigInt(100000000), // 100 USDC
    nonce: BigInt(1),
    expiry: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
    policyId: 'default-spending-policy',
    policyVersion: 1,
  };

  const validProof: SpendingProof = {
    proofHash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    inputHash: '0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd',
    modelHash: '0x7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b',
    decision: {
      shouldBuy: true,
      confidence: 0.92,
      riskScore: 0.15,
    },
    timestamp: Date.now(),
    proofSizeBytes: 48000,
    generationTimeMs: 2100,
    verified: true,
    txIntentHash: computeTxIntentHash(validIntent),
  };

  beforeEach(() => {
    resetContractState();
  });

  describe('computeTxIntentHash', () => {
    it('should compute deterministic hash for same intent', () => {
      const hash1 = computeTxIntentHash(validIntent);
      const hash2 = computeTxIntentHash(validIntent);

      expect(hash1).toBe(hash2);
      expect(hash1).toMatch(/^0x[0-9a-f]{64}$/);
    });

    it('should compute different hashes for different intents', () => {
      const hash1 = computeTxIntentHash(validIntent);
      const hash2 = computeTxIntentHash({
        ...validIntent,
        amount: BigInt(200000000),
      });

      expect(hash1).not.toBe(hash2);
    });
  });

  describe('gatedTransfer', () => {
    it('should reject transfer without proof', async () => {
      const result = await gatedTransfer(validIntent, null);

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('PROOF_REQUIRED');
    });

    it('should reject transfer when proof decision is false', async () => {
      const rejectedProof: SpendingProof = {
        ...validProof,
        decision: {
          shouldBuy: false,
          confidence: 0.1,
          riskScore: 0.9,
        },
      };

      const result = await gatedTransfer(validIntent, rejectedProof);

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('POLICY_REJECTED');
    });

    it('should reject when amount is modified', async () => {
      const result = await gatedTransfer(validIntent, validProof, { modifyAmount: true });

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('INTENT_MISMATCH');
    });

    it('should reject replay attacks', async () => {
      // First transfer should succeed
      const result1 = await gatedTransfer(validIntent, validProof);
      expect(result1.success).toBe(true);

      // Second transfer with same nonce should fail
      const result2 = await gatedTransfer(validIntent, validProof, { replayNonce: true });
      expect(result2.success).toBe(false);
      expect(result2.revertReason).toContain('NONCE_ALREADY_USED');
    });

    it('should reject expired proofs', async () => {
      const expiredIntent: TxIntent = {
        ...validIntent,
        expiry: Math.floor(Date.now() / 1000) - 3600, // 1 hour ago
        nonce: BigInt(999), // Different nonce to avoid replay check
      };
      const unboundProof = { ...validProof, txIntentHash: undefined };

      const result = await gatedTransfer(expiredIntent, unboundProof);

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('PROOF_EXPIRED');
    });

    it('should reject unknown policy', async () => {
      const unknownPolicyIntent: TxIntent = {
        ...validIntent,
        policyId: 'unknown-policy',
        nonce: BigInt(888),
      };
      // Proof without txIntentHash binding to bypass the hash check
      // (in production, proof would be generated per-intent)
      const unboundProof = { ...validProof, txIntentHash: undefined };

      const result = await gatedTransfer(unknownPolicyIntent, unboundProof);

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('UNKNOWN_POLICY');
    });

    it('should reject version mismatch', async () => {
      const wrongVersionIntent: TxIntent = {
        ...validIntent,
        policyVersion: 2, // Default policy is version 1
        nonce: BigInt(777),
      };
      const unboundProof = { ...validProof, txIntentHash: undefined };

      const result = await gatedTransfer(wrongVersionIntent, unboundProof);

      expect(result.success).toBe(false);
      expect(result.revertReason).toContain('VERSION_MISMATCH');
    });

    it('should succeed with valid proof and intent', async () => {
      const result = await gatedTransfer(validIntent, validProof);

      expect(result.success).toBe(true);
      expect(result.txHash).toMatch(/^0x[0-9a-f]{64}$/);
      expect(result.gasUsed).toBeGreaterThan(0);
    });
  });

  describe('Policy Registry', () => {
    it('should check if default policy is registered', () => {
      expect(isPolicyRegistered('default-spending-policy')).toBe(true);
      expect(isPolicyRegistered('nonexistent-policy')).toBe(false);
    });

    it('should register new policies', () => {
      const newPolicyId = 'custom-policy';
      expect(isPolicyRegistered(newPolicyId)).toBe(false);

      registerPolicy(
        newPolicyId,
        '0xmodel',
        '0xvk',
        1
      );

      expect(isPolicyRegistered(newPolicyId)).toBe(true);
    });
  });
});
