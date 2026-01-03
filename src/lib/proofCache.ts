/**
 * In-memory cache for proof generation results
 *
 * Since the same inputs always produce the same proof, we can cache
 * proof results to avoid regenerating proofs for identical requests.
 */

import { keccak256, toBytes } from 'viem';
import type { ProveResponse } from './types';

export interface CacheEntry {
  response: ProveResponse;
  timestamp: number;
  hits: number;
}

export interface CacheStats {
  size: number;
  maxSize: number;
  hits: number;
  misses: number;
  hitRate: number;
}

export interface ProofCacheOptions {
  /** Maximum number of entries to cache (default: 100) */
  maxSize?: number;
  /** Time-to-live in milliseconds (default: 15 minutes) */
  ttlMs?: number;
  /** Whether caching is enabled (default: true) */
  enabled?: boolean;
}

const DEFAULT_OPTIONS: Required<ProofCacheOptions> = {
  maxSize: 100,
  ttlMs: 15 * 60 * 1000, // 15 minutes
  enabled: true,
};

/**
 * Create a cache key from inputs and tag
 */
export function createCacheKey(inputs: number[], tag: string): string {
  const data = JSON.stringify({ inputs, tag });
  return keccak256(toBytes(data));
}

/**
 * Proof cache implementation with LRU eviction
 */
export class ProofCache {
  private cache: Map<string, CacheEntry> = new Map();
  private readonly options: Required<ProofCacheOptions>;
  private hits = 0;
  private misses = 0;

  constructor(options: ProofCacheOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Get a cached proof response
   */
  get(inputs: number[], tag: string): ProveResponse | null {
    if (!this.options.enabled) {
      return null;
    }

    const key = createCacheKey(inputs, tag);
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return null;
    }

    // Check if expired
    if (Date.now() - entry.timestamp > this.options.ttlMs) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    // Update hit count and move to end (LRU)
    entry.hits++;
    this.cache.delete(key);
    this.cache.set(key, entry);

    this.hits++;
    return entry.response;
  }

  /**
   * Store a proof response in the cache
   */
  set(inputs: number[], tag: string, response: ProveResponse): void {
    if (!this.options.enabled) {
      return;
    }

    // Only cache successful responses
    if (!response.success) {
      return;
    }

    const key = createCacheKey(inputs, tag);

    // Evict oldest entries if at capacity
    while (this.cache.size >= this.options.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(key, {
      response,
      timestamp: Date.now(),
      hits: 0,
    });
  }

  /**
   * Check if a proof is cached
   */
  has(inputs: number[], tag: string): boolean {
    if (!this.options.enabled) {
      return false;
    }

    const key = createCacheKey(inputs, tag);
    const entry = this.cache.get(key);

    if (!entry) {
      return false;
    }

    // Check if expired
    if (Date.now() - entry.timestamp > this.options.ttlMs) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Clear all cached entries
   */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Remove expired entries
   */
  prune(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.options.ttlMs) {
        this.cache.delete(key);
        pruned++;
      }
    }

    return pruned;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      maxSize: this.options.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  /**
   * Enable or disable the cache
   */
  setEnabled(enabled: boolean): void {
    this.options.enabled = enabled;
    if (!enabled) {
      this.clear();
    }
  }

  /**
   * Check if cache is enabled
   */
  isEnabled(): boolean {
    return this.options.enabled;
  }
}

// Global cache instance
let globalCache: ProofCache | null = null;

/**
 * Get the global proof cache instance
 */
export function getProofCache(options?: ProofCacheOptions): ProofCache {
  if (!globalCache) {
    globalCache = new ProofCache(options);
  }
  return globalCache;
}

/**
 * Reset the global cache (useful for testing)
 */
export function resetProofCache(): void {
  globalCache?.clear();
  globalCache = null;
}

/**
 * Wrapper function to add caching to proof generation
 */
export async function withProofCache<T extends ProveResponse>(
  inputs: number[],
  tag: string,
  generateFn: () => Promise<T>,
  cache: ProofCache = getProofCache()
): Promise<T> {
  // Check cache first
  const cached = cache.get(inputs, tag);
  if (cached) {
    return cached as T;
  }

  // Generate proof
  const result = await generateFn();

  // Cache successful results
  if (result.success) {
    cache.set(inputs, tag, result);
  }

  return result;
}
