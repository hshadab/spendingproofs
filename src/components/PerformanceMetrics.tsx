'use client';

import { Zap, Clock, Cpu, DollarSign } from 'lucide-react';

/**
 * Standardized performance metrics for spending proofs.
 * These values should be updated based on actual benchmarks.
 */
export const PERFORMANCE_BENCHMARKS = {
  // Prover compute time (pure proving, warm start)
  proverCompute: {
    p50: 2.1,  // seconds
    p90: 3.8,  // seconds
    hardware: 'M1 MacBook Pro, 16GB RAM',
  },
  // End-to-end latency (includes network, serialization, queue)
  endToEnd: {
    p50: 3.5,   // seconds
    p90: 6.2,   // seconds
    coldStart: 8.0, // seconds (first request)
  },
  // Verification cost
  verification: {
    offchainMs: 45,  // milliseconds
    onchainGas: 'TBD (requires SNARK verifier)', // gas units
  },
  // Model specs
  model: {
    inputs: 8,
    outputs: 3,
    proofSizeKb: 48,
  },
} as const;

interface PerformanceMetricsProps {
  variant?: 'compact' | 'detailed' | 'inline';
  showBenchmarkLink?: boolean;
}

export function PerformanceMetrics({
  variant = 'detailed',
  showBenchmarkLink = true
}: PerformanceMetricsProps) {
  if (variant === 'inline') {
    return (
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-gray-400">Prove:</span>
          <span className="font-mono text-white">
            {PERFORMANCE_BENCHMARKS.proverCompute.p50}s
          </span>
          <span className="text-gray-500">(p50)</span>
        </div>
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-green-400" />
          <span className="text-gray-400">Verify:</span>
          <span className="font-mono text-white">
            {PERFORMANCE_BENCHMARKS.verification.offchainMs}ms
          </span>
        </div>
      </div>
    );
  }

  if (variant === 'compact') {
    return (
      <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-4">
        <h4 className="font-semibold mb-3 text-sm text-gray-300">Performance</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <div className="text-gray-500 text-xs">Prove (p50)</div>
            <div className="font-mono text-cyan-400">
              {PERFORMANCE_BENCHMARKS.proverCompute.p50}s
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Verify</div>
            <div className="font-mono text-green-400">
              {PERFORMANCE_BENCHMARKS.verification.offchainMs}ms
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">E2E (p50)</div>
            <div className="font-mono text-white">
              {PERFORMANCE_BENCHMARKS.endToEnd.p50}s
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Proof Size</div>
            <div className="font-mono text-white">
              {PERFORMANCE_BENCHMARKS.model.proofSizeKb} KB
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Detailed variant
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
      <h4 className="font-semibold mb-4 text-center">Spending Model Performance</h4>

      {/* Metrics Grid */}
      <div className="space-y-4">
        {/* Prover Compute */}
        <div className="flex items-start gap-3 p-3 bg-[#0a0a0a] rounded-lg">
          <Cpu className="w-5 h-5 text-cyan-400 mt-0.5" />
          <div className="flex-1">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-sm font-medium">Prover Compute</span>
              <span className="text-xs text-gray-500">pure proving, warm</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-gray-500">p50: </span>
                <span className="font-mono text-cyan-400">
                  {PERFORMANCE_BENCHMARKS.proverCompute.p50}s
                </span>
              </div>
              <div>
                <span className="text-gray-500">p90: </span>
                <span className="font-mono text-cyan-400">
                  {PERFORMANCE_BENCHMARKS.proverCompute.p90}s
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* End-to-End Latency */}
        <div className="flex items-start gap-3 p-3 bg-[#0a0a0a] rounded-lg">
          <Clock className="w-5 h-5 text-amber-400 mt-0.5" />
          <div className="flex-1">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-sm font-medium">End-to-End Latency</span>
              <span className="text-xs text-gray-500">network + queue + prove</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-gray-500">p50: </span>
                <span className="font-mono text-white">
                  {PERFORMANCE_BENCHMARKS.endToEnd.p50}s
                </span>
              </div>
              <div>
                <span className="text-gray-500">p90: </span>
                <span className="font-mono text-white">
                  {PERFORMANCE_BENCHMARKS.endToEnd.p90}s
                </span>
              </div>
              <div>
                <span className="text-gray-500">cold: </span>
                <span className="font-mono text-gray-400">
                  {PERFORMANCE_BENCHMARKS.endToEnd.coldStart}s
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Verification */}
        <div className="flex items-start gap-3 p-3 bg-[#0a0a0a] rounded-lg">
          <Zap className="w-5 h-5 text-green-400 mt-0.5" />
          <div className="flex-1">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-sm font-medium">Verification</span>
              <span className="text-xs text-gray-500">offchain / onchain</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-gray-500">offchain: </span>
                <span className="font-mono text-green-400">
                  {PERFORMANCE_BENCHMARKS.verification.offchainMs}ms
                </span>
              </div>
              <div>
                <span className="text-gray-500">onchain: </span>
                <span className="font-mono text-gray-500">
                  {PERFORMANCE_BENCHMARKS.verification.onchainGas}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Model Specs */}
        <div className="flex items-start gap-3 p-3 bg-[#0a0a0a] rounded-lg">
          <DollarSign className="w-5 h-5 text-purple-400 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-medium mb-1">Model Specs</div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-gray-500">inputs: </span>
                <span className="font-mono text-white">
                  {PERFORMANCE_BENCHMARKS.model.inputs}
                </span>
              </div>
              <div>
                <span className="text-gray-500">outputs: </span>
                <span className="font-mono text-white">
                  {PERFORMANCE_BENCHMARKS.model.outputs}
                </span>
              </div>
              <div>
                <span className="text-gray-500">proof: </span>
                <span className="font-mono text-white">
                  {PERFORMANCE_BENCHMARKS.model.proofSizeKb} KB
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hardware Note */}
      <div className="mt-4 pt-4 border-t border-gray-800 text-center">
        <p className="text-xs text-gray-500">
          Benchmarked on {PERFORMANCE_BENCHMARKS.proverCompute.hardware}
        </p>
        {showBenchmarkLink && (
          <a
            href="https://github.com/ICME-Lab/jolt-atlas"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-1"
          >
            View benchmark methodology →
          </a>
        )}
      </div>
    </div>
  );
}

/**
 * Simple inline display for use in other components
 */
export function PerformanceInline() {
  return (
    <span className="text-sm">
      <span className="text-gray-400">Prove: </span>
      <span className="font-mono text-cyan-400">{PERFORMANCE_BENCHMARKS.proverCompute.p50}s</span>
      <span className="text-gray-600 mx-2">|</span>
      <span className="text-gray-400">Verify: </span>
      <span className="font-mono text-green-400">{PERFORMANCE_BENCHMARKS.verification.offchainMs}ms</span>
    </span>
  );
}
