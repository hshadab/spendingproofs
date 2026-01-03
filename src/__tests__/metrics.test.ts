import { describe, it, expect, beforeEach } from 'vitest';
import {
  ProofMetricsCollector,
  getMetricsCollector,
  resetMetrics,
  StructuredLogger,
  createLogger,
} from '@/lib/metrics';

describe('metrics', () => {
  describe('ProofMetricsCollector', () => {
    let collector: ProofMetricsCollector;

    beforeEach(() => {
      collector = new ProofMetricsCollector();
    });

    it('should start with zero metrics', () => {
      const metrics = collector.getMetrics();

      expect(metrics.requestsTotal).toBe(0);
      expect(metrics.successTotal).toBe(0);
      expect(metrics.failureTotal).toBe(0);
      expect(metrics.cacheHits).toBe(0);
      expect(metrics.cacheMisses).toBe(0);
      expect(metrics.retriesTotal).toBe(0);
      expect(metrics.avgGenerationTimeMs).toBe(0);
      expect(metrics.p95GenerationTimeMs).toBe(0);
      expect(metrics.maxGenerationTimeMs).toBe(0);
    });

    it('should record requests', () => {
      collector.recordRequest();
      collector.recordRequest();
      collector.recordRequest();

      expect(collector.getMetrics().requestsTotal).toBe(3);
    });

    it('should record success with generation time', () => {
      collector.recordSuccess(5000);
      collector.recordSuccess(6000);
      collector.recordSuccess(7000);

      const metrics = collector.getMetrics();
      expect(metrics.successTotal).toBe(3);
      expect(metrics.avgGenerationTimeMs).toBe(6000);
      expect(metrics.maxGenerationTimeMs).toBe(7000);
    });

    it('should record failures', () => {
      collector.recordFailure();
      collector.recordFailure();

      expect(collector.getMetrics().failureTotal).toBe(2);
    });

    it('should record cache hits and misses', () => {
      collector.recordCacheHit();
      collector.recordCacheHit();
      collector.recordCacheMiss();

      const metrics = collector.getMetrics();
      expect(metrics.cacheHits).toBe(2);
      expect(metrics.cacheMisses).toBe(1);
    });

    it('should record retries', () => {
      collector.recordRetry();
      collector.recordRetry();
      collector.recordRetry();

      expect(collector.getMetrics().retriesTotal).toBe(3);
    });

    it('should calculate success rate', () => {
      collector.recordRequest();
      collector.recordRequest();
      collector.recordRequest();
      collector.recordRequest();
      collector.recordSuccess(1000);
      collector.recordSuccess(1000);
      collector.recordSuccess(1000);
      collector.recordFailure();

      expect(collector.getSuccessRate()).toBe(0.75);
    });

    it('should return 0 success rate when no requests', () => {
      expect(collector.getSuccessRate()).toBe(0);
    });

    it('should calculate cache hit rate', () => {
      collector.recordCacheHit();
      collector.recordCacheHit();
      collector.recordCacheMiss();
      collector.recordCacheMiss();

      expect(collector.getCacheHitRate()).toBe(0.5);
    });

    it('should return 0 cache hit rate when no cache operations', () => {
      expect(collector.getCacheHitRate()).toBe(0);
    });

    it('should calculate p95 generation time', () => {
      // Add 100 samples
      for (let i = 1; i <= 100; i++) {
        collector.recordSuccess(i * 100);
      }

      const metrics = collector.getMetrics();
      // P95 should be around 9500-9600 (95th percentile of 100-10000)
      expect(metrics.p95GenerationTimeMs).toBeGreaterThanOrEqual(9500);
      expect(metrics.p95GenerationTimeMs).toBeLessThanOrEqual(9600);
    });

    it('should limit number of samples', () => {
      const collector = new ProofMetricsCollector({ maxSamples: 10 });

      for (let i = 0; i < 20; i++) {
        collector.recordSuccess(1000);
      }

      expect(collector.getMetrics().sampleCount).toBe(10);
    });

    it('should reset all metrics', () => {
      collector.recordRequest();
      collector.recordSuccess(5000);
      collector.recordFailure();
      collector.recordCacheHit();
      collector.recordRetry();

      collector.reset();

      const metrics = collector.getMetrics();
      expect(metrics.requestsTotal).toBe(0);
      expect(metrics.successTotal).toBe(0);
      expect(metrics.failureTotal).toBe(0);
      expect(metrics.cacheHits).toBe(0);
      expect(metrics.retriesTotal).toBe(0);
    });

    it('should export Prometheus format', () => {
      collector.recordRequest();
      collector.recordSuccess(5000);

      const prometheus = collector.toPrometheusFormat();

      expect(prometheus).toContain('spending_proofs_requests_total 1');
      expect(prometheus).toContain('spending_proofs_success_total 1');
      expect(prometheus).toContain('# TYPE spending_proofs_requests_total counter');
    });

    it('should use custom prefix in Prometheus format', () => {
      const prometheus = collector.toPrometheusFormat('custom_prefix');

      expect(prometheus).toContain('custom_prefix_requests_total');
      expect(prometheus).toContain('custom_prefix_success_total');
    });
  });

  describe('getMetricsCollector', () => {
    beforeEach(() => {
      resetMetrics();
    });

    it('should return the same instance', () => {
      const collector1 = getMetricsCollector();
      const collector2 = getMetricsCollector();

      expect(collector1).toBe(collector2);
    });

    it('should persist data across calls', () => {
      const collector = getMetricsCollector();
      collector.recordRequest();

      expect(getMetricsCollector().getMetrics().requestsTotal).toBe(1);
    });
  });

  describe('resetMetrics', () => {
    it('should reset global metrics', () => {
      const collector = getMetricsCollector();
      collector.recordRequest();
      collector.recordSuccess(5000);

      resetMetrics();

      expect(getMetricsCollector().getMetrics().requestsTotal).toBe(0);
    });
  });

  describe('StructuredLogger', () => {
    it('should log with correct level', () => {
      const logger = new StructuredLogger('test', true);

      // These should not throw
      expect(() => logger.debug('debug message')).not.toThrow();
      expect(() => logger.info('info message')).not.toThrow();
      expect(() => logger.warn('warn message')).not.toThrow();
      expect(() => logger.error('error message')).not.toThrow();
    });

    it('should include context in logs', () => {
      const logger = new StructuredLogger('test', true);

      // Should not throw with context
      expect(() =>
        logger.info('message', { action: 'test', duration: 100 })
      ).not.toThrow();
    });

    it('should create child loggers', () => {
      const parent = new StructuredLogger('parent', true);
      const child = parent.child('child');

      expect(child).toBeInstanceOf(StructuredLogger);
    });

    it('should not log when disabled', () => {
      const logger = new StructuredLogger('test', false);

      // Should not throw even when disabled
      expect(() => logger.info('message')).not.toThrow();
    });
  });

  describe('createLogger', () => {
    it('should create a logger with component name', () => {
      const logger = createLogger('TestComponent');

      expect(logger).toBeInstanceOf(StructuredLogger);
    });
  });
});
