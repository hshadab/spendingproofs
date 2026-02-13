import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createProveMessage,
  verifyProveRequest,
  isSignatureAuthEnabled,
} from '@/lib/signatureAuth';

// Mock viem's verifyMessage
vi.mock('viem', async () => {
  const actual = await vi.importActual('viem');
  return {
    ...actual,
    verifyMessage: vi.fn(),
  };
});

import { verifyMessage } from 'viem';

describe('signatureAuth', () => {
  const mockAddress = '0x1234567890123456789012345678901234567890' as `0x${string}`;
  const mockSignature = '0xabcd' as `0x${string}`;
  const mockInputs = [0.05, 1.0, 0.2, 0.5, 0.95, 100, 5, 2.5];
  const mockTag = 'spending';

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(verifyMessage).mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('createProveMessage', () => {
    it('should create a formatted message', () => {
      const timestamp = 1704067200000;
      const message = createProveMessage(mockInputs, mockTag, timestamp);

      expect(message).toContain('Spending Proofs Authentication');
      expect(message).toContain('Action: Generate proof');
      expect(message).toContain(`Tag: ${mockTag}`);
      expect(message).toContain('Input Hash: 0x');
      expect(message).toContain(`Timestamp: ${timestamp}`);
    });

    it('should create deterministic messages', () => {
      const timestamp = 1704067200000;
      const message1 = createProveMessage(mockInputs, mockTag, timestamp);
      const message2 = createProveMessage(mockInputs, mockTag, timestamp);

      expect(message1).toBe(message2);
    });

    it('should create different messages for different inputs', () => {
      const timestamp = 1704067200000;
      const message1 = createProveMessage([1, 2, 3], 'tag1', timestamp);
      const message2 = createProveMessage([1, 2, 4], 'tag1', timestamp);
      const message3 = createProveMessage([1, 2, 3], 'tag2', timestamp);

      expect(message1).not.toBe(message2);
      expect(message1).not.toBe(message3);
    });
  });

  describe('verifyProveRequest', () => {
    it('should return valid for correct signature', async () => {
      vi.mocked(verifyMessage).mockResolvedValue(true);
      const now = Date.now();

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: now,
        signature: mockSignature,
      });

      expect(result.valid).toBe(true);
      expect(result.address).toBe(mockAddress);
    });

    it('should reject expired signatures', async () => {
      const now = Date.now();
      const oldTimestamp = now - (6 * 60 * 1000); // 6 minutes ago (beyond 5 min expiry)

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: oldTimestamp,
        signature: mockSignature,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('SIGNATURE_EXPIRED');
      expect(result.error).toContain('Signature expired');
    });

    it('should reject future timestamps', async () => {
      const now = Date.now();
      const futureTimestamp = now + (60 * 1000); // 1 minute in future (beyond 10s tolerance)

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: futureTimestamp,
        signature: mockSignature,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('CLOCK_SKEW');
      expect(result.error).toContain('future');
    });

    it('should reject invalid signatures', async () => {
      vi.mocked(verifyMessage).mockResolvedValue(false);
      const now = Date.now();

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: now,
        signature: mockSignature,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_SIGNATURE');
      expect(result.error).toContain('Invalid signature');
    });

    it('should handle verification errors gracefully', async () => {
      vi.mocked(verifyMessage).mockRejectedValue(new Error('Verification error'));
      const now = Date.now();

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: now,
        signature: mockSignature,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_SIGNATURE');
      expect(result.error).toContain('Verification error');
    });

    it('should verify with correct message', async () => {
      vi.mocked(verifyMessage).mockResolvedValue(true);
      const now = Date.now();

      await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: now,
        signature: mockSignature,
      });

      expect(verifyMessage).toHaveBeenCalledWith({
        address: mockAddress,
        message: expect.stringContaining('Spending Proofs Authentication'),
        signature: mockSignature,
      });
    });

    it('should allow slight clock skew (within 10s)', async () => {
      vi.mocked(verifyMessage).mockResolvedValue(true);
      const now = Date.now();
      const slightlyFuture = now + 5000; // 5 seconds in future (within 10s tolerance)

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: slightlyFuture,
        signature: mockSignature,
      });

      expect(result.valid).toBe(true);
    });

    it('should reject timestamps beyond clock skew tolerance', async () => {
      const now = Date.now();
      const beyondTolerance = now + 15000; // 15 seconds in future (beyond 10s tolerance)

      const result = await verifyProveRequest({
        inputs: mockInputs,
        tag: mockTag,
        address: mockAddress,
        timestamp: beyondTolerance,
        signature: mockSignature,
      });

      expect(result.valid).toBe(false);
      expect(result.code).toBe('CLOCK_SKEW');
    });
  });

  describe('isSignatureAuthEnabled', () => {
    const originalEnv = process.env;

    beforeEach(() => {
      process.env = { ...originalEnv };
    });

    afterEach(() => {
      process.env = originalEnv;
    });

    it('should return false when env is not set', () => {
      delete process.env.REQUIRE_SIGNATURE_AUTH;
      expect(isSignatureAuthEnabled()).toBe(false);
    });

    it('should return false when env is "false"', () => {
      process.env.REQUIRE_SIGNATURE_AUTH = 'false';
      expect(isSignatureAuthEnabled()).toBe(false);
    });

    it('should return true when env is "true"', () => {
      process.env.REQUIRE_SIGNATURE_AUTH = 'true';
      expect(isSignatureAuthEnabled()).toBe(true);
    });
  });
});
