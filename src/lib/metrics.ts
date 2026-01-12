/**
 * Metrics and observability utilities for Spending Proofs
 *
 * This module provides lightweight metrics collection for monitoring
 * proof generation performance and error rates.
 */

export interface MetricPoint {
  timestamp: number;
  value: number;
  labels?: Record<string, string>;
}

export interface ProofMetrics {
  /** Total number of proof generation requests */
  requestsTotal: number;
  /** Number of successful proof generations */
  successTotal: number;
  /** Number of failed proof generations */
  failureTotal: number;
  /** Number of cache hits */
  cacheHits: number;
  /** Number of cache misses */
  cacheMisses: number;
  /** Number of retries performed */
  retriesTotal: number;
  /** Average proof generation time in ms */
  avgGenerationTimeMs: number;
  /** 95th percentile generation time in ms */
  p95GenerationTimeMs: number;
  /** Max generation time in ms */
  maxGenerationTimeMs: number;
  /** Generation times for percentile calculations */
  generationTimes: number[];
}

const DEFAULT_METRICS: ProofMetrics = {
  requestsTotal: 0,
  successTotal: 0,
  failureTotal: 0,
  cacheHits: 0,
  cacheMisses: 0,
  retriesTotal: 0,
  avgGenerationTimeMs: 0,
  p95GenerationTimeMs: 0,
  maxGenerationTimeMs: 0,
  generationTimes: [],
};

// Maximum number of generation times to keep for percentile calculations
const MAX_GENERATION_TIMES = 1000;

/**
 * Metrics collector for proof generation
 */
export class ProofMetricsCollector {
  private metrics: ProofMetrics = { ...DEFAULT_METRICS, generationTimes: [] };
  private readonly maxSamples: number;

  constructor(options: { maxSamples?: number } = {}) {
    this.maxSamples = options.maxSamples ?? MAX_GENERATION_TIMES;
  }

  /**
   * Record a proof generation request
   */
  recordRequest(): void {
    this.metrics.requestsTotal++;
  }

  /**
   * Record a successful proof generation
   */
  recordSuccess(generationTimeMs: number): void {
    this.metrics.successTotal++;
    this.addGenerationTime(generationTimeMs);
  }

  /**
   * Record a failed proof generation
   */
  recordFailure(): void {
    this.metrics.failureTotal++;
  }

  /**
   * Record a cache hit
   */
  recordCacheHit(): void {
    this.metrics.cacheHits++;
  }

  /**
   * Record a cache miss
   */
  recordCacheMiss(): void {
    this.metrics.cacheMisses++;
  }

  /**
   * Record a retry attempt
   */
  recordRetry(): void {
    this.metrics.retriesTotal++;
  }

  /**
   * Add a generation time sample
   */
  private addGenerationTime(timeMs: number): void {
    this.metrics.generationTimes.push(timeMs);

    // Keep only the most recent samples
    if (this.metrics.generationTimes.length > this.maxSamples) {
      this.metrics.generationTimes.shift();
    }

    // Update max
    if (timeMs > this.metrics.maxGenerationTimeMs) {
      this.metrics.maxGenerationTimeMs = timeMs;
    }

    // Recalculate average and p95
    this.recalculateStats();
  }

  /**
   * Recalculate average and percentile statistics
   */
  private recalculateStats(): void {
    const times = this.metrics.generationTimes;
    if (times.length === 0) return;

    // Average
    const sum = times.reduce((a, b) => a + b, 0);
    this.metrics.avgGenerationTimeMs = Math.round(sum / times.length);

    // P95
    const sorted = [...times].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);
    this.metrics.p95GenerationTimeMs = sorted[p95Index] ?? sorted[sorted.length - 1];
  }

  /**
   * Get current metrics
   */
  getMetrics(): Omit<ProofMetrics, 'generationTimes'> & { sampleCount: number } {
    const { generationTimes, ...rest } = this.metrics;
    return {
      ...rest,
      sampleCount: generationTimes.length,
    };
  }

  /**
   * Get success rate (0-1)
   */
  getSuccessRate(): number {
    if (this.metrics.requestsTotal === 0) return 0;
    return this.metrics.successTotal / this.metrics.requestsTotal;
  }

  /**
   * Get cache hit rate (0-1)
   */
  getCacheHitRate(): number {
    const total = this.metrics.cacheHits + this.metrics.cacheMisses;
    if (total === 0) return 0;
    return this.metrics.cacheHits / total;
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.metrics = { ...DEFAULT_METRICS, generationTimes: [] };
  }

  /**
   * Export metrics in Prometheus format
   */
  toPrometheusFormat(prefix = 'spending_proofs'): string {
    const metrics = this.getMetrics();
    const lines: string[] = [];

    lines.push(`# HELP ${prefix}_requests_total Total number of proof generation requests`);
    lines.push(`# TYPE ${prefix}_requests_total counter`);
    lines.push(`${prefix}_requests_total ${metrics.requestsTotal}`);

    lines.push(`# HELP ${prefix}_success_total Number of successful proof generations`);
    lines.push(`# TYPE ${prefix}_success_total counter`);
    lines.push(`${prefix}_success_total ${metrics.successTotal}`);

    lines.push(`# HELP ${prefix}_failure_total Number of failed proof generations`);
    lines.push(`# TYPE ${prefix}_failure_total counter`);
    lines.push(`${prefix}_failure_total ${metrics.failureTotal}`);

    lines.push(`# HELP ${prefix}_cache_hits_total Number of cache hits`);
    lines.push(`# TYPE ${prefix}_cache_hits_total counter`);
    lines.push(`${prefix}_cache_hits_total ${metrics.cacheHits}`);

    lines.push(`# HELP ${prefix}_retries_total Number of retry attempts`);
    lines.push(`# TYPE ${prefix}_retries_total counter`);
    lines.push(`${prefix}_retries_total ${metrics.retriesTotal}`);

    lines.push(`# HELP ${prefix}_generation_time_avg_ms Average proof generation time`);
    lines.push(`# TYPE ${prefix}_generation_time_avg_ms gauge`);
    lines.push(`${prefix}_generation_time_avg_ms ${metrics.avgGenerationTimeMs}`);

    lines.push(`# HELP ${prefix}_generation_time_p95_ms 95th percentile generation time`);
    lines.push(`# TYPE ${prefix}_generation_time_p95_ms gauge`);
    lines.push(`${prefix}_generation_time_p95_ms ${metrics.p95GenerationTimeMs}`);

    return lines.join('\n');
  }
}

// Global metrics instance
let globalMetrics: ProofMetricsCollector | null = null;

/**
 * Get the global metrics collector
 */
export function getMetricsCollector(): ProofMetricsCollector {
  if (!globalMetrics) {
    globalMetrics = new ProofMetricsCollector();
  }
  return globalMetrics;
}

/**
 * Reset the global metrics collector
 */
export function resetMetrics(): void {
  globalMetrics?.reset();
}

/**
 * Logging utilities with structured output
 */
export interface LogContext {
  component?: string;
  action?: string;
  duration?: number;
  error?: unknown;
  [key: string]: unknown;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Structured logger for consistent log output
 */
export class StructuredLogger {
  private readonly component: string;
  private readonly enabled: boolean;

  constructor(component: string, enabled = true) {
    this.component = component;
    this.enabled = enabled;
  }

  private log(level: LogLevel, message: string, context?: LogContext): void {
    if (!this.enabled) return;

    const entry = {
      timestamp: new Date().toISOString(),
      level,
      component: this.component,
      message,
      ...context,
    };

    // Remove undefined values
    Object.keys(entry).forEach((key) => {
      if (entry[key as keyof typeof entry] === undefined) {
        delete entry[key as keyof typeof entry];
      }
    });

    // Custom replacer to handle BigInt and Error objects
    const replacer = (_key: string, value: unknown): unknown => {
      if (typeof value === 'bigint') {
        return value.toString();
      }
      if (value instanceof Error) {
        return { message: value.message, name: value.name };
      }
      return value;
    };

    const logFn = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
    logFn(JSON.stringify(entry, replacer));
  }

  debug(message: string, context?: LogContext): void {
    this.log('debug', message, context);
  }

  info(message: string, context?: LogContext): void {
    this.log('info', message, context);
  }

  warn(message: string, context?: LogContext): void {
    this.log('warn', message, context);
  }

  error(message: string, context?: LogContext): void {
    this.log('error', message, context);
  }

  /**
   * Create a child logger with additional context
   */
  child(component: string): StructuredLogger {
    return new StructuredLogger(`${this.component}:${component}`, this.enabled);
  }
}

/**
 * Create a logger for a component
 */
export function createLogger(component: string): StructuredLogger {
  const enabled = typeof process !== 'undefined'
    ? process.env.NODE_ENV !== 'test'
    : true;
  return new StructuredLogger(component, enabled);
}
