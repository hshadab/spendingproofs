import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  withRetry,
  calculateDelay,
  isRetryableError,
  createRetryWrapper,
  proverRetryOptions,
} from '@/lib/retry';

describe('retry utilities', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('calculateDelay', () => {
    it('should calculate exponential backoff', () => {
      const options = { initialDelayMs: 1000, backoffMultiplier: 2, jitter: 0 };

      // attempt 0: 1000 * 2^0 = 1000
      expect(calculateDelay(0, { ...options, maxDelayMs: 10000 })).toBe(1000);

      // attempt 1: 1000 * 2^1 = 2000
      expect(calculateDelay(1, { ...options, maxDelayMs: 10000 })).toBe(2000);

      // attempt 2: 1000 * 2^2 = 4000
      expect(calculateDelay(2, { ...options, maxDelayMs: 10000 })).toBe(4000);
    });

    it('should cap at maxDelayMs', () => {
      const options = {
        initialDelayMs: 1000,
        backoffMultiplier: 2,
        maxDelayMs: 5000,
        jitter: 0,
      };

      // attempt 3: 1000 * 2^3 = 8000, capped at 5000
      expect(calculateDelay(3, options)).toBe(5000);
    });

    it('should add jitter within range', () => {
      const options = {
        initialDelayMs: 1000,
        backoffMultiplier: 1,
        maxDelayMs: 10000,
        jitter: 0.1,
      };

      // Run multiple times to verify jitter
      for (let i = 0; i < 10; i++) {
        const delay = calculateDelay(0, options);
        expect(delay).toBeGreaterThanOrEqual(900);
        expect(delay).toBeLessThanOrEqual(1100);
      }
    });
  });

  describe('isRetryableError', () => {
    it('should return true for network errors', () => {
      expect(isRetryableError(new Error('fetch failed'))).toBe(true);
      expect(isRetryableError(new Error('network error'))).toBe(true);
      expect(isRetryableError(new Error('connection refused'))).toBe(true);
      expect(isRetryableError(new Error('timeout'))).toBe(true);
      expect(isRetryableError(new Error('ECONNREFUSED'))).toBe(true);
      expect(isRetryableError(new Error('ENOTFOUND'))).toBe(true);
    });

    it('should return true for 5xx errors', () => {
      expect(isRetryableError(new Error('Server error: 500'))).toBe(true);
      expect(isRetryableError(new Error('502 Bad Gateway'))).toBe(true);
      expect(isRetryableError(new Error('503 Service Unavailable'))).toBe(true);
      expect(isRetryableError(new Error('504 Gateway Timeout'))).toBe(true);
    });

    it('should return true for prover unavailable', () => {
      expect(isRetryableError(new Error('prover unavailable'))).toBe(true);
    });

    it('should return false for non-retryable errors', () => {
      expect(isRetryableError(new Error('Invalid input'))).toBe(false);
      expect(isRetryableError(new Error('Validation failed'))).toBe(false);
      expect(isRetryableError(new Error('Some random error'))).toBe(false);
    });

    it('should return false for non-Error values', () => {
      expect(isRetryableError('string error')).toBe(false);
      expect(isRetryableError(null)).toBe(false);
      expect(isRetryableError(undefined)).toBe(false);
    });
  });

  describe('withRetry', () => {
    it('should return success on first try', async () => {
      const fn = vi.fn().mockResolvedValue('success');

      const resultPromise = withRetry(fn, { maxAttempts: 3 });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(true);
      expect(result.data).toBe('success');
      expect(result.attempts).toBe(1);
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should retry on retryable errors', async () => {
      const fn = vi
        .fn()
        .mockRejectedValueOnce(new Error('network error'))
        .mockRejectedValueOnce(new Error('timeout'))
        .mockResolvedValue('success');

      const resultPromise = withRetry(fn, {
        maxAttempts: 3,
        initialDelayMs: 100,
        jitter: 0,
      });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(true);
      expect(result.data).toBe('success');
      expect(result.attempts).toBe(3);
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should fail after max attempts', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('network error'));

      const resultPromise = withRetry(fn, {
        maxAttempts: 3,
        initialDelayMs: 100,
        jitter: 0,
      });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.attempts).toBe(3);
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should not retry on non-retryable errors', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('Validation failed'));

      const resultPromise = withRetry(fn, { maxAttempts: 3 });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(false);
      expect(result.attempts).toBe(3); // Still shows max attempts
      expect(fn).toHaveBeenCalledTimes(1); // But only called once
    });

    it('should call onRetry callback', async () => {
      const fn = vi
        .fn()
        .mockRejectedValueOnce(new Error('network error'))
        .mockResolvedValue('success');
      const onRetry = vi.fn();

      const resultPromise = withRetry(fn, {
        maxAttempts: 3,
        initialDelayMs: 100,
        jitter: 0,
        onRetry,
      });
      await vi.runAllTimersAsync();
      await resultPromise;

      expect(onRetry).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledWith(expect.any(Error), 1, 100);
    });

    it('should respect abort signal', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('network error'));
      const controller = new AbortController();

      setTimeout(() => controller.abort(), 50);

      const resultPromise = withRetry(fn, {
        maxAttempts: 5,
        initialDelayMs: 100,
        signal: controller.signal,
      });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(false);
      expect(fn).toHaveBeenCalled();
    });
  });

  describe('createRetryWrapper', () => {
    it('should create a wrapper with default options', async () => {
      const wrapper = createRetryWrapper({ maxAttempts: 2, jitter: 0 });
      const fn = vi.fn().mockResolvedValue('result');

      const resultPromise = wrapper(fn);
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(true);
      expect(result.data).toBe('result');
    });

    it('should allow overriding options', async () => {
      const wrapper = createRetryWrapper({ maxAttempts: 2 });
      const fn = vi.fn().mockRejectedValue(new Error('network error'));

      const resultPromise = wrapper(fn, { maxAttempts: 1 });
      await vi.runAllTimersAsync();
      const result = await resultPromise;

      expect(result.success).toBe(false);
      expect(fn).toHaveBeenCalledTimes(1);
    });
  });

  describe('proverRetryOptions', () => {
    it('should have correct default values', () => {
      expect(proverRetryOptions.maxAttempts).toBe(3);
      expect(proverRetryOptions.initialDelayMs).toBe(2000);
      expect(proverRetryOptions.maxDelayMs).toBe(10000);
      expect(proverRetryOptions.backoffMultiplier).toBe(1.5);
      expect(proverRetryOptions.jitter).toBe(0.2);
    });

    it('should not retry validation errors', () => {
      const isRetryable = proverRetryOptions.isRetryable!;
      expect(isRetryable(new Error('invalid input'), 0)).toBe(false);
      expect(isRetryable(new Error('validation failed'), 0)).toBe(false);
      expect(isRetryable(new Error('unauthorized'), 0)).toBe(false);
      expect(isRetryable(new Error('forbidden'), 0)).toBe(false);
    });

    it('should retry network errors', () => {
      const isRetryable = proverRetryOptions.isRetryable!;
      expect(isRetryable(new Error('network error'), 0)).toBe(true);
      expect(isRetryable(new Error('timeout'), 0)).toBe(true);
    });
  });
});
