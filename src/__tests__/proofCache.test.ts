import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ProofCache,
  createCacheKey,
  getProofCache,
  resetProofCache,
  withProofCache,
} from '@/lib/proofCache';
import type { ProveResponse } from '@/lib/types';

describe('proofCache', () => {
  beforeEach(() => {
    resetProofCache();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const mockSuccessResponse: ProveResponse = {
    success: true,
    proof: {
      proof: '0x123',
      proofHash: '0xabc',
      programIo: '0xdef',
      metadata: {
        modelHash: '0x111',
        inputHash: '0x222',
        outputHash: '0x333',
        proofSize: 48000,
        generationTime: 5000,
        proverVersion: 'jolt-atlas-v1.0.0',
      },
      tag: 'spending',
      timestamp: Date.now(),
    },
    inference: {
      output: 1,
      rawOutput: [1, 0.95, 0.1],
      decision: 'approve',
      confidence: 0.95,
    },
    generationTimeMs: 5000,
  };

  const mockFailedResponse: ProveResponse = {
    success: false,
    error: 'Generation failed',
    generationTimeMs: 100,
  };

  describe('createCacheKey', () => {
    it('should create deterministic keys', () => {
      const inputs = [0.05, 1.0, 0.2, 0.5, 0.95, 100, 5, 2.5];
      const tag = 'spending';

      const key1 = createCacheKey(inputs, tag);
      const key2 = createCacheKey(inputs, tag);

      expect(key1).toBe(key2);
      expect(key1).toMatch(/^0x[a-f0-9]{64}$/);
    });

    it('should create different keys for different inputs', () => {
      const key1 = createCacheKey([1, 2, 3], 'tag1');
      const key2 = createCacheKey([1, 2, 4], 'tag1');
      const key3 = createCacheKey([1, 2, 3], 'tag2');

      expect(key1).not.toBe(key2);
      expect(key1).not.toBe(key3);
      expect(key2).not.toBe(key3);
    });
  });

  describe('ProofCache', () => {
    it('should store and retrieve cached responses', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);
      const result = cache.get(inputs, tag);

      expect(result).toEqual(mockSuccessResponse);
    });

    it('should return null for cache miss', () => {
      const cache = new ProofCache();
      const result = cache.get([1, 2, 3], 'test');

      expect(result).toBeNull();
    });

    it('should not cache failed responses', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockFailedResponse);
      const result = cache.get(inputs, tag);

      expect(result).toBeNull();
    });

    it('should respect TTL', () => {
      const cache = new ProofCache({ ttlMs: 1000 });
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);

      // Still valid
      vi.advanceTimersByTime(500);
      expect(cache.get(inputs, tag)).toEqual(mockSuccessResponse);

      // Expired
      vi.advanceTimersByTime(600);
      expect(cache.get(inputs, tag)).toBeNull();
    });

    it('should evict oldest entries when at capacity', () => {
      const cache = new ProofCache({ maxSize: 2 });

      cache.set([1], 'a', mockSuccessResponse);
      cache.set([2], 'b', mockSuccessResponse);
      cache.set([3], 'c', mockSuccessResponse);

      // Oldest entry should be evicted
      expect(cache.get([1], 'a')).toBeNull();
      expect(cache.get([2], 'b')).toEqual(mockSuccessResponse);
      expect(cache.get([3], 'c')).toEqual(mockSuccessResponse);
    });

    it('should update hit count on access', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);

      // Access multiple times
      cache.get(inputs, tag);
      cache.get(inputs, tag);
      cache.get(inputs, tag);

      const stats = cache.getStats();
      expect(stats.hits).toBe(3);
    });

    it('should track misses', () => {
      const cache = new ProofCache();

      cache.get([1], 'a');
      cache.get([2], 'b');

      const stats = cache.getStats();
      expect(stats.misses).toBe(2);
    });

    it('should calculate hit rate correctly', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);

      // 3 hits
      cache.get(inputs, tag);
      cache.get(inputs, tag);
      cache.get(inputs, tag);

      // 1 miss
      cache.get([9, 9, 9], 'missing');

      const stats = cache.getStats();
      expect(stats.hitRate).toBe(0.75); // 3 / 4
    });

    it('should check if entry exists with has()', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      expect(cache.has(inputs, tag)).toBe(false);

      cache.set(inputs, tag, mockSuccessResponse);
      expect(cache.has(inputs, tag)).toBe(true);
    });

    it('should clear all entries', () => {
      const cache = new ProofCache();

      cache.set([1], 'a', mockSuccessResponse);
      cache.set([2], 'b', mockSuccessResponse);

      cache.clear();

      expect(cache.get([1], 'a')).toBeNull();
      expect(cache.get([2], 'b')).toBeNull();
      expect(cache.getStats().size).toBe(0);
    });

    it('should prune expired entries', () => {
      const cache = new ProofCache({ ttlMs: 1000 });

      cache.set([1], 'a', mockSuccessResponse);
      vi.advanceTimersByTime(500);
      cache.set([2], 'b', mockSuccessResponse);
      vi.advanceTimersByTime(600);

      const pruned = cache.prune();

      expect(pruned).toBe(1); // First entry expired
      expect(cache.get([1], 'a')).toBeNull();
      expect(cache.get([2], 'b')).toEqual(mockSuccessResponse);
    });

    it('should respect enabled flag', () => {
      const cache = new ProofCache({ enabled: false });
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);
      expect(cache.get(inputs, tag)).toBeNull();
      expect(cache.has(inputs, tag)).toBe(false);
    });

    it('should toggle enabled state', () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';

      cache.set(inputs, tag, mockSuccessResponse);
      expect(cache.isEnabled()).toBe(true);

      cache.setEnabled(false);
      expect(cache.isEnabled()).toBe(false);
      expect(cache.get(inputs, tag)).toBeNull();

      cache.setEnabled(true);
      cache.set(inputs, tag, mockSuccessResponse);
      expect(cache.get(inputs, tag)).toEqual(mockSuccessResponse);
    });
  });

  describe('getProofCache', () => {
    it('should return the same instance', () => {
      const cache1 = getProofCache();
      const cache2 = getProofCache();

      expect(cache1).toBe(cache2);
    });

    it('should respect options on first call', () => {
      const cache = getProofCache({ maxSize: 50 });
      const stats = cache.getStats();

      expect(stats.maxSize).toBe(50);
    });
  });

  describe('withProofCache', () => {
    it('should return cached result on hit', async () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';
      const generateFn = vi.fn().mockResolvedValue(mockSuccessResponse);

      // First call - generates
      const result1 = await withProofCache(inputs, tag, generateFn, cache);
      expect(result1).toEqual(mockSuccessResponse);
      expect(generateFn).toHaveBeenCalledTimes(1);

      // Second call - cached
      const result2 = await withProofCache(inputs, tag, generateFn, cache);
      expect(result2).toEqual(mockSuccessResponse);
      expect(generateFn).toHaveBeenCalledTimes(1); // Not called again
    });

    it('should not cache failed results', async () => {
      const cache = new ProofCache();
      const inputs = [1, 2, 3];
      const tag = 'test';
      const generateFn = vi.fn().mockResolvedValue(mockFailedResponse);

      await withProofCache(inputs, tag, generateFn, cache);
      await withProofCache(inputs, tag, generateFn, cache);

      expect(generateFn).toHaveBeenCalledTimes(2);
    });

    it('should use global cache by default', async () => {
      const inputs = [1, 2, 3];
      const tag = 'test';
      const generateFn = vi.fn().mockResolvedValue(mockSuccessResponse);

      await withProofCache(inputs, tag, generateFn);
      await withProofCache(inputs, tag, generateFn);

      expect(generateFn).toHaveBeenCalledTimes(1);
    });
  });
});
