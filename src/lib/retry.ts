/**
 * Retry utilities with exponential backoff
 */

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

  // Add jitter: random value between -jitter% and +jitter%
  const jitterRange = cappedDelay * jitter;
  const jitterValue = (Math.random() * 2 - 1) * jitterRange;

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
 * Default retry check - retries on network errors and 5xx responses
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();

    // Network errors
    if (
      msg.includes('fetch') ||
      msg.includes('network') ||
      msg.includes('connection') ||
      msg.includes('timeout') ||
      msg.includes('econnrefused') ||
      msg.includes('enotfound')
    ) {
      return true;
    }

    // Server errors (5xx)
    if (msg.includes('500') || msg.includes('502') || msg.includes('503') || msg.includes('504')) {
      return true;
    }

    // Prover unavailable
    if (msg.includes('prover') && msg.includes('unavailable')) {
      return true;
    }
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
