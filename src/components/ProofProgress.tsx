'use client';

import { useEffect, useState } from 'react';
import { Loader2, Check, Clock, AlertCircle } from 'lucide-react';
import type { ProofStep, ProofStatus } from '@/lib/types';

interface ProofProgressProps {
  status: ProofStatus;
  progress: number;
  elapsedMs: number;
  steps: ProofStep[];
}

const DEFAULT_STEPS: ProofStep[] = [
  { name: 'Running ONNX inference', status: 'pending' },
  { name: 'Preparing witness data', status: 'pending' },
  { name: 'Generating JOLT SNARK proof', status: 'pending' },
  { name: 'Computing commitments', status: 'pending' },
  { name: 'Finalizing proof', status: 'pending' },
];

export function ProofProgress({ status, progress, elapsedMs, steps }: ProofProgressProps) {
  const [displaySteps, setDisplaySteps] = useState<ProofStep[]>(DEFAULT_STEPS);

  useEffect(() => {
    if (steps.length > 0) {
      setDisplaySteps(steps);
    } else if (status === 'running') {
      // Simulate step progression
      const stepProgress = Math.floor((progress / 100) * 5);
      setDisplaySteps(
        DEFAULT_STEPS.map((step, i) => ({
          ...step,
          status: i < stepProgress ? 'done' : i === stepProgress ? 'running' : 'pending',
        }))
      );
    } else if (status === 'complete') {
      setDisplaySteps(DEFAULT_STEPS.map((step) => ({ ...step, status: 'done' })));
    } else if (status === 'error') {
      setDisplaySteps(DEFAULT_STEPS.map((step, i) => ({
        ...step,
        status: i === 0 ? 'error' : 'pending',
      })));
    } else {
      setDisplaySteps(DEFAULT_STEPS);
    }
  }, [status, progress, steps]);

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const estimatedTotal = 8000; // 8 seconds estimate
  const estimatedRemaining = Math.max(0, estimatedTotal - elapsedMs);

  if (status === 'idle') {
    return null;
  }

  return (
    <div className="bg-[#0d1117] rounded-xl border border-gray-800 border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white text-white">
          {status === 'running' && 'Generating zkML Proof'}
          {status === 'complete' && 'Proof Generated'}
          {status === 'error' && 'Proof Generation Failed'}
        </h3>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Clock className="w-4 h-4" />
          {formatTime(elapsedMs)}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="h-3 bg-slate-100 bg-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              status === 'error'
                ? 'bg-red-500'
                : status === 'complete'
                ? 'bg-green-500'
                : 'bg-purple-600'
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-sm mt-2">
          <span className="text-gray-500">{progress.toFixed(0)}%</span>
          {status === 'running' && (
            <span className="text-gray-400">
              ~{formatTime(estimatedRemaining)} remaining
            </span>
          )}
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {displaySteps.map((step, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center">
              {step.status === 'done' && (
                <div className="w-5 h-5 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                  <Check className="w-3 h-3 text-green-600 text-green-400" />
                </div>
              )}
              {step.status === 'running' && (
                <Loader2 className="w-5 h-5 text-purple-600 animate-spin" />
              )}
              {step.status === 'pending' && (
                <div className="w-3 h-3 rounded-full bg-gray-700 bg-700" />
              )}
              {step.status === 'error' && (
                <div className="w-5 h-5 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                  <AlertCircle className="w-3 h-3 text-red-600 text-red-400" />
                </div>
              )}
            </div>
            <span
              className={`text-sm ${
                step.status === 'done'
                  ? 'text-white text-white'
                  : step.status === 'running'
                  ? 'text-purple-600 text-purple-400 font-medium'
                  : step.status === 'error'
                  ? 'text-red-600 text-red-400'
                  : 'text-gray-400'
              }`}
            >
              {step.name}
            </span>
            {step.durationMs !== undefined && (
              <span className="text-xs text-gray-400 ml-auto">
                {formatTime(step.durationMs)}
              </span>
            )}
          </div>
        ))}
      </div>

    </div>
  );
}
