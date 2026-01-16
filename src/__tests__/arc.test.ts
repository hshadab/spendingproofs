import { describe, it, expect } from 'vitest';
import {
  formatProofHash,
  getExplorerUrl,
  getContractExplorerUrl,
} from '../lib/arc';

describe('Arc Integration Utilities', () => {
  describe('formatProofHash', () => {
    it('should return hash unchanged if already valid bytes32', () => {
      const hash = '0x' + 'a'.repeat(64);
      const result = formatProofHash(hash);
      expect(result).toBe(hash);
    });

    it('should add 0x prefix if missing', () => {
      const hash = 'a'.repeat(64);
      const result = formatProofHash(hash);
      expect(result).toBe('0x' + hash);
    });

    it('should pad short hash to 32 bytes', () => {
      const shortHash = '0x1234';
      const result = formatProofHash(shortHash);
      expect(result).toHaveLength(66); // 0x + 64 chars
      expect(result.startsWith('0x')).toBe(true);
      expect(result.endsWith('1234')).toBe(true);
    });

    it('should pad hash without 0x prefix', () => {
      const shortHash = 'abcd';
      const result = formatProofHash(shortHash);
      expect(result).toHaveLength(66);
      expect(result.startsWith('0x')).toBe(true);
    });

    it('should handle empty string', () => {
      const result = formatProofHash('');
      expect(result).toBe('0x' + '0'.repeat(64));
    });

    it('should handle 0x only', () => {
      const result = formatProofHash('0x');
      expect(result).toBe('0x' + '0'.repeat(64));
    });
  });

  describe('getExplorerUrl', () => {
    it('should generate correct explorer URL for transaction', () => {
      const txHash = '0x1234567890abcdef';
      const result = getExplorerUrl(txHash);
      expect(result).toBe(`https://testnet.arcscan.app/tx/${txHash}`);
    });

    it('should handle hash without 0x prefix', () => {
      const txHash = '1234567890abcdef';
      const result = getExplorerUrl(txHash);
      expect(result).toBe(`https://testnet.arcscan.app/tx/${txHash}`);
    });
  });

  describe('getContractExplorerUrl', () => {
    it('should generate correct explorer URL for contract address', () => {
      const address = '0x1234567890123456789012345678901234567890';
      const result = getContractExplorerUrl(address);
      expect(result).toBe(`https://testnet.arcscan.app/address/${address}`);
    });

    it('should handle address without 0x prefix', () => {
      const address = '1234567890123456789012345678901234567890';
      const result = getContractExplorerUrl(address);
      expect(result).toBe(`https://testnet.arcscan.app/address/${address}`);
    });
  });
});
