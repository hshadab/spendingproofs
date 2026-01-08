/**
 * Retry utilities with exponential backoff
 */

import { generateSecureJitter } from './crypto';

export interface RetryOptions {
  /** Maximum number of retry attempts (default: 3) */
  maxAttempts?: number;
  /** Initial delay in milliseconds (default: 1000) */
  initialDelayMs?: number;
  /** Maximum delay in milliseconds (default: 30000) */
  maxDelayMs?: number;
  /** Backoff multiplier (default: 2) */
  backoffMultiplier?: number;
  /** Jitter factor 0-1 to add randomness (default: 0.1) */
  jitter?: number;
  /** Function to determine if error is retryable */
  isRetryable?: (error: unknown, attempt: number) => boolean;
  /** Callback on each retry attempt */
  onRetry?: (error: unknown, attempt: number, delayMs: number) => void;
  /** AbortSignal for cancellation */
  signal?: AbortSignal;
}

const DEFAULT_OPTIONS: Required<Omit<RetryOptions, 'signal' | 'onRetry' | 'isRetryable'>> = {
  maxAttempts: 3,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffMultiplier: 2,
  jitter: 0.1,
};

/**
 * Calculate delay for a given attempt with exponential backoff and jitter
 */
export function calculateDelay(
  attempt: number,
  options: Pick<RetryOptions, 'initialDelayMs' | 'maxDelayMs' | 'backoffMultiplier' | 'jitter'>
): number {
  const {
    initialDelayMs = DEFAULT_OPTIONS.initialDelayMs,
    maxDelayMs = DEFAULT_OPTIONS.maxDelayMs,
    backoffMultiplier = DEFAULT_OPTIONS.backoffMultiplier,
    jitter = DEFAULT_OPTIONS.jitter,
  } = options;

  // Exponential backoff: initialDelay * multiplier^attempt
  const exponentialDelay = initialDelayMs * Math.pow(backoffMultiplier, attempt);

  // Cap at max delay
  const cappedDelay = Math.min(exponentialDelay, maxDelayMs);

  // Add jitter using cryptographically secure randomness
  const jitterRange = cappedDelay * jitter;
  const jitterValue = generateSecureJitter(jitterRange);

  return Math.round(cappedDelay + jitterValue);
}

/**
 * Sleep for a given number of milliseconds
 */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error('Aborted'));
      return;
    }

    const timeoutId = setTimeout(resolve, ms);

    signal?.addEventListener('abort', () => {
      clearTimeout(timeoutId);
      reject(new Error('Aborted'));
    });
  });
}

/**
 * Known retryable error types
 */
export const RetryableErrorTypes = {
  NETWORK: 'NETWORK',
  SERVER_ERROR: 'SERVER_ERROR',
  TIMEOUT: 'TIMEOUT',
  PROVER_UNAVAILABLE: 'PROVER_UNAVAILABLE',
} as const;

/**
 * Check if an error is a network-related error by type, not string matching
 */
function isNetworkError(error: unknown): boolean {
  // TypeError is thrown by fetch for network failures
  if (error instanceof TypeError) {
    return true;
  }

  // Check for DOMException (AbortError, NetworkError)
  if (error instanceof DOMException) {
    return error.name === 'NetworkError' || error.name === 'AbortError';
  }

  // Check for Node.js system errors
  if (error instanceof Error) {
    const nodeError = error as Error & { code?: string };
    const networkErrorCodes = [
      'ECONNREFUSED',
      'ECONNRESET',
      'ENOTFOUND',
      'ETIMEDOUT',
      'ENETUNREACH',
      'EHOSTUNREACH',
      'EPIPE',
      'EAI_AGAIN',
    ];
    if (nodeError.code && networkErrorCodes.includes(nodeError.code)) {
      return true;
    }
  }

  return false;
}

/**
 * Check if an error indicates a server-side issue (5xx)
 */
function isServerError(error: unknown): boolean {
  if (error instanceof Error) {
    const httpError = error as Error & { status?: number; statusCode?: number };
    const status = httpError.status || httpError.statusCode;
    if (status && status >= 500 && status < 600) {
      return true;
    }
  }
  return false;
}

/**
 * Check if an error indicates the prover is unavailable
 */
function isProverUnavailableError(error: unknown): boolean {
  if (error instanceof Error) {
    const proverError = error as Error & { code?: string };
    return proverError.code === 'PROVER_UNAVAILABLE';
  }
  return false;
}

/**
 * Default retry check - retries on network errors, timeouts, and 5xx responses
 * Uses error types instead of fragile string matching
 */
export function isRetryableError(error: unknown): boolean {
  // Network errors (fetch failures, connection issues)
  if (isNetworkError(error)) {
    return true;
  }

  // Server errors (5xx)
  if (isServerError(error)) {
    return true;
  }

  // Prover unavailable
  if (isProverUnavailableError(error)) {
    return true;
  }

  return false;
}

export interface RetryResult<T> {
  success: boolean;
  data?: T;
  error?: unknown;
  attempts: number;
  totalTimeMs: number;
}

/**
 * Execute a function with retry logic
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<RetryResult<T>> {
  const {
    maxAttempts = DEFAULT_OPTIONS.maxAttempts,
    initialDelayMs = DEFAULT_OPTIONS.initialDelayMs,
    maxDelayMs = DEFAULT_OPTIONS.maxDelayMs,
    backoffMultiplier = DEFAULT_OPTIONS.backoffMultiplier,
    jitter = DEFAULT_OPTIONS.jitter,
    isRetryable = isRetryableError,
    onRetry,
    signal,
  } = options;

  const startTime = Date.now();
  let lastError: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      // Check if aborted
      if (signal?.aborted) {
        throw new Error('Aborted');
      }

      const data = await fn();
      return {
        success: true,
        data,
        attempts: attempt + 1,
        totalTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      lastError = error;

      // Check if we should retry
      const shouldRetry = attempt < maxAttempts - 1 && isRetryable(error, attempt);

      if (!shouldRetry) {
        break;
      }

      // Calculate delay
      const delay = calculateDelay(attempt, {
        initialDelayMs,
        maxDelayMs,
        backoffMultiplier,
        jitter,
      });

      // Notify about retry
      onRetry?.(error, attempt + 1, delay);

      // Wait before retrying
      try {
        await sleep(delay, signal);
      } catch {
        // Aborted during sleep
        break;
      }
    }
  }

  return {
    success: false,
    error: lastError,
    attempts: maxAttempts,
    totalTimeMs: Date.now() - startTime,
  };
}

/**
 * Create a retry wrapper with preset options
 */
export function createRetryWrapper(defaultOptions: RetryOptions) {
  return <T>(fn: () => Promise<T>, overrides?: RetryOptions): Promise<RetryResult<T>> => {
    return withRetry(fn, { ...defaultOptions, ...overrides });
  };
}

/**
 * Prover-specific retry configuration
 */
export const proverRetryOptions: RetryOptions = {
  maxAttempts: 3,
  initialDelayMs: 2000,
  maxDelayMs: 10000,
  backoffMultiplier: 1.5,
  jitter: 0.2,
  isRetryable: (error) => {
    // Always retry on prover unavailable
    if (isRetryableError(error)) {
      return true;
    }

    // Don't retry on validation errors
    if (error instanceof Error) {
      const msg = error.message.toLowerCase();
      if (
        msg.includes('invalid') ||
        msg.includes('validation') ||
        msg.includes('unauthorized') ||
        msg.includes('forbidden')
      ) {
        return false;
      }
    }

    return false;
  },
};
