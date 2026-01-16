import { describe, it, expect } from 'vitest';
import {
  validateAddress,
  validatePositiveNumber,
  validateProofHash,
  validateString,
  validateTransferInput,
  validateWalletInput,
  createValidationErrorResponse,
  createErrorResponse,
} from '../lib/validation';

describe('Validation Utilities', () => {
  describe('validateAddress', () => {
    it('should accept valid Ethereum addresses', () => {
      const result = validateAddress('0x1234567890123456789012345678901234567890', 'to');
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should accept checksummed addresses', () => {
      const result = validateAddress('0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B', 'to');
      expect(result.valid).toBe(true);
    });

    it('should reject undefined address', () => {
      const result = validateAddress(undefined, 'to');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('to is required');
    });

    it('should reject null address', () => {
      const result = validateAddress(null, 'to');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('to is required');
    });

    it('should reject non-string address', () => {
      const result = validateAddress(12345, 'to');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('to must be a string');
    });

    it('should reject invalid address format', () => {
      const result = validateAddress('not-an-address', 'to');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('to is not a valid Ethereum address');
    });

    it('should reject address without 0x prefix', () => {
      const result = validateAddress('1234567890123456789012345678901234567890', 'to');
      expect(result.valid).toBe(false);
    });

    it('should reject address with wrong length', () => {
      const result = validateAddress('0x123', 'to');
      expect(result.valid).toBe(false);
    });
  });

  describe('validatePositiveNumber', () => {
    it('should accept positive numbers', () => {
      const result = validatePositiveNumber(1.5, 'amount');
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should accept positive string numbers', () => {
      const result = validatePositiveNumber('1.5', 'amount');
      expect(result.valid).toBe(true);
    });

    it('should reject zero', () => {
      const result = validatePositiveNumber(0, 'amount');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be positive');
    });

    it('should reject negative numbers', () => {
      const result = validatePositiveNumber(-1, 'amount');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be positive');
    });

    it('should reject undefined', () => {
      const result = validatePositiveNumber(undefined, 'amount');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount is required');
    });

    it('should reject NaN', () => {
      const result = validatePositiveNumber(NaN, 'amount');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be a valid number');
    });

    it('should reject non-numeric strings', () => {
      const result = validatePositiveNumber('abc', 'amount');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be a valid number');
    });

    it('should respect min option', () => {
      const result = validatePositiveNumber(0.5, 'amount', { min: 1 });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be at least 1');
    });

    it('should respect max option', () => {
      const result = validatePositiveNumber(100, 'amount', { max: 50 });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('amount must be at most 50');
    });

    it('should accept values within min/max range', () => {
      const result = validatePositiveNumber(25, 'amount', { min: 10, max: 50 });
      expect(result.valid).toBe(true);
    });
  });

  describe('validateProofHash', () => {
    it('should accept valid bytes32 hash', () => {
      const hash = '0x' + 'a'.repeat(64);
      const result = validateProofHash(hash, 'proofHash', true);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should accept mixed case hex', () => {
      const hash = '0xAaBbCcDd' + 'e'.repeat(56);
      const result = validateProofHash(hash, 'proofHash', true);
      expect(result.valid).toBe(true);
    });

    it('should accept empty value when not required', () => {
      const result = validateProofHash('', 'proofHash', false);
      expect(result.valid).toBe(true);
    });

    it('should accept undefined when not required', () => {
      const result = validateProofHash(undefined, 'proofHash', false);
      expect(result.valid).toBe(true);
    });

    it('should reject empty value when required', () => {
      const result = validateProofHash('', 'proofHash', true);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('proofHash is required');
    });

    it('should reject hash without 0x prefix', () => {
      const hash = 'a'.repeat(64);
      const result = validateProofHash(hash, 'proofHash', true);
      expect(result.valid).toBe(false);
    });

    it('should reject hash with wrong length', () => {
      const hash = '0x' + 'a'.repeat(32);
      const result = validateProofHash(hash, 'proofHash', true);
      expect(result.valid).toBe(false);
    });

    it('should reject hash with invalid hex characters', () => {
      const hash = '0x' + 'g'.repeat(64);
      const result = validateProofHash(hash, 'proofHash', true);
      expect(result.valid).toBe(false);
    });

    it('should reject non-string value', () => {
      const result = validateProofHash(123, 'proofHash', true);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('proofHash must be a string');
    });
  });

  describe('validateString', () => {
    it('should accept valid string', () => {
      const result = validateString('hello', 'name');
      expect(result.valid).toBe(true);
    });

    it('should reject empty string when required', () => {
      const result = validateString('', 'name', { required: true });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('name is required');
    });

    it('should accept empty string when not required', () => {
      const result = validateString('', 'name', { required: false });
      expect(result.valid).toBe(true);
    });

    it('should reject non-string value', () => {
      const result = validateString(123, 'name');
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('name must be a string');
    });

    it('should respect minLength', () => {
      const result = validateString('ab', 'name', { minLength: 3 });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('name must be at least 3 characters');
    });

    it('should respect maxLength', () => {
      const result = validateString('abcdefgh', 'name', { maxLength: 5 });
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('name must be at most 5 characters');
    });

    it('should accept string within length constraints', () => {
      const result = validateString('abcd', 'name', { minLength: 2, maxLength: 5 });
      expect(result.valid).toBe(true);
    });
  });

  describe('validateTransferInput', () => {
    const validInput = {
      to: '0x1234567890123456789012345678901234567890',
      amount: 1.5,
    };

    it('should accept valid transfer input', () => {
      const result = validateTransferInput(validInput);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should accept transfer with valid proofHash', () => {
      const input = {
        ...validInput,
        proofHash: '0x' + 'a'.repeat(64),
      };
      const result = validateTransferInput(input);
      expect(result.valid).toBe(true);
    });

    it('should reject missing to address', () => {
      const result = validateTransferInput({ amount: 1.5 });
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('to'))).toBe(true);
    });

    it('should reject invalid to address', () => {
      const result = validateTransferInput({ to: 'invalid', amount: 1.5 });
      expect(result.valid).toBe(false);
    });

    it('should reject missing amount', () => {
      const result = validateTransferInput({ to: validInput.to });
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('amount'))).toBe(true);
    });

    it('should reject zero amount', () => {
      const result = validateTransferInput({ ...validInput, amount: 0 });
      expect(result.valid).toBe(false);
    });

    it('should reject amount below minimum', () => {
      const result = validateTransferInput({ ...validInput, amount: 0.0000001 });
      expect(result.valid).toBe(false);
    });

    it('should reject amount above maximum', () => {
      const result = validateTransferInput({ ...validInput, amount: 2000000 });
      expect(result.valid).toBe(false);
    });

    it('should reject invalid proofHash format', () => {
      const input = {
        ...validInput,
        proofHash: 'invalid-hash',
      };
      const result = validateTransferInput(input);
      expect(result.valid).toBe(false);
    });

    it('should collect multiple errors', () => {
      const result = validateTransferInput({ to: 'invalid', amount: -1 });
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(1);
    });
  });

  describe('validateWalletInput', () => {
    it('should accept empty input', () => {
      const result = validateWalletInput({});
      expect(result.valid).toBe(true);
    });

    it('should accept valid userId', () => {
      const result = validateWalletInput({ userId: 'user123' });
      expect(result.valid).toBe(true);
    });

    it('should accept undefined userId', () => {
      const result = validateWalletInput({ userId: undefined });
      expect(result.valid).toBe(true);
    });

    it('should reject userId exceeding max length', () => {
      const result = validateWalletInput({ userId: 'a'.repeat(101) });
      expect(result.valid).toBe(false);
    });

    it('should reject non-string userId', () => {
      const result = validateWalletInput({ userId: 123 });
      expect(result.valid).toBe(false);
    });
  });

  describe('Error Response Helpers', () => {
    describe('createValidationErrorResponse', () => {
      it('should create response with single error', () => {
        const response = createValidationErrorResponse(['Field is required']);
        expect(response.success).toBe(false);
        expect(response.error).toBe('Field is required');
        expect(response.code).toBe('VALIDATION_ERROR');
        expect(response.details).toBeUndefined();
      });

      it('should create response with multiple errors', () => {
        const errors = ['Error 1', 'Error 2', 'Error 3'];
        const response = createValidationErrorResponse(errors);
        expect(response.success).toBe(false);
        expect(response.error).toBe('Validation failed');
        expect(response.code).toBe('VALIDATION_ERROR');
        expect(response.details).toEqual(errors);
      });
    });

    describe('createErrorResponse', () => {
      it('should create generic error response', () => {
        const response = createErrorResponse('Something went wrong', 'INTERNAL_ERROR');
        expect(response.success).toBe(false);
        expect(response.error).toBe('Something went wrong');
        expect(response.code).toBe('INTERNAL_ERROR');
        expect(response.details).toBeUndefined();
      });
    });
  });
});
