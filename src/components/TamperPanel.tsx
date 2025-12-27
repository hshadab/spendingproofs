'use client';

import { useState } from 'react';
import { Check, X, AlertTriangle, Shield, Edit3 } from 'lucide-react';
import { spendingInputToNumeric, type SpendingModelInput } from '@/lib/spendingModel';
import { truncateHash } from '@/lib/contracts';

interface TamperPanelProps {
  originalInputs: SpendingModelInput;
  proofInputHash: string;
  onVerify: (inputs: number[], isModified: boolean) => Promise<{ valid: boolean; reason: string }>;
}

export function TamperPanel({ originalInputs, proofInputHash, onVerify }: TamperPanelProps) {
  const [modifiedInputs, setModifiedInputs] = useState<SpendingModelInput>(originalInputs);
  const [originalResult, setOriginalResult] = useState<{ valid: boolean; reason: string } | null>(null);
  const [modifiedResult, setModifiedResult] = useState<{ valid: boolean; reason: string } | null>(null);
  const [verifying, setVerifying] = useState<'original' | 'modified' | null>(null);

  const handleVerifyOriginal = async () => {
    setVerifying('original');
    const result = await onVerify(spendingInputToNumeric(originalInputs), false);
    setOriginalResult(result);
    setVerifying(null);
  };

  const handleVerifyModified = async () => {
    setVerifying('modified');
    const result = await onVerify(spendingInputToNumeric(modifiedInputs), true);
    setModifiedResult(result);
    setVerifying(null);
  };

  const hasModifications =
    modifiedInputs.priceUsdc !== originalInputs.priceUsdc ||
    modifiedInputs.budgetUsdc !== originalInputs.budgetUsdc ||
    modifiedInputs.spentTodayUsdc !== originalInputs.spentTodayUsdc;

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
      <div className="flex items-center gap-2 mb-6">
        <AlertTriangle className="w-5 h-5 text-amber-500" />
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
          Tamper Detection
        </h3>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Original Values */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-green-600" />
            <span className="font-medium text-slate-900 dark:text-white">
              Original Values (Proof Generated)
            </span>
          </div>

          <div className="space-y-3 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Price</span>
              <span className="font-mono text-slate-900 dark:text-white">
                ${originalInputs.priceUsdc.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Budget</span>
              <span className="font-mono text-slate-900 dark:text-white">
                ${originalInputs.budgetUsdc.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Spent Today</span>
              <span className="font-mono text-slate-900 dark:text-white">
                ${originalInputs.spentTodayUsdc.toFixed(2)}
              </span>
            </div>
          </div>

          <button
            onClick={handleVerifyOriginal}
            disabled={verifying !== null}
            className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {verifying === 'original' ? (
              <>Verifying...</>
            ) : (
              <>
                <Check className="w-4 h-4" />
                Verify Original
              </>
            )}
          </button>

          {originalResult && (
            <div
              className={`p-3 rounded-lg flex items-center gap-2 ${
                originalResult.valid
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
              }`}
            >
              {originalResult.valid ? (
                <Check className="w-4 h-4 flex-shrink-0" />
              ) : (
                <X className="w-4 h-4 flex-shrink-0" />
              )}
              <span className="text-sm">{originalResult.reason}</span>
            </div>
          )}
        </div>

        {/* Modified Values */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Edit3 className="w-4 h-4 text-amber-500" />
            <span className="font-medium text-slate-900 dark:text-white">
              Modified Values (Try to Tamper)
            </span>
          </div>

          <div className="space-y-3 p-4 bg-amber-50 dark:bg-amber-900/10 rounded-lg border border-amber-200 dark:border-amber-800">
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-500">Price</span>
              <input
                type="number"
                step="0.001"
                value={modifiedInputs.priceUsdc}
                onChange={(e) =>
                  setModifiedInputs({ ...modifiedInputs, priceUsdc: parseFloat(e.target.value) || 0 })
                }
                className="w-24 px-2 py-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-right font-mono text-sm"
              />
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-500">Budget</span>
              <input
                type="number"
                step="0.01"
                value={modifiedInputs.budgetUsdc}
                onChange={(e) =>
                  setModifiedInputs({ ...modifiedInputs, budgetUsdc: parseFloat(e.target.value) || 0 })
                }
                className="w-24 px-2 py-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-right font-mono text-sm"
              />
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-slate-500">Spent Today</span>
              <input
                type="number"
                step="0.01"
                value={modifiedInputs.spentTodayUsdc}
                onChange={(e) =>
                  setModifiedInputs({ ...modifiedInputs, spentTodayUsdc: parseFloat(e.target.value) || 0 })
                }
                className="w-24 px-2 py-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-right font-mono text-sm"
              />
            </div>
          </div>

          <button
            onClick={handleVerifyModified}
            disabled={verifying !== null}
            className={`w-full py-2 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${
              hasModifications
                ? 'bg-amber-600 hover:bg-amber-700 disabled:bg-amber-400 text-white'
                : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
            }`}
          >
            {verifying === 'modified' ? (
              <>Verifying...</>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4" />
                Verify Modified
              </>
            )}
          </button>

          {modifiedResult && (
            <div
              className={`p-3 rounded-lg flex items-center gap-2 ${
                modifiedResult.valid
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
              }`}
            >
              {modifiedResult.valid ? (
                <Check className="w-4 h-4 flex-shrink-0" />
              ) : (
                <X className="w-4 h-4 flex-shrink-0" />
              )}
              <span className="text-sm">{modifiedResult.reason}</span>
            </div>
          )}
        </div>
      </div>

      {/* Proof Hash */}
      <div className="mt-6 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
        <div className="text-xs text-slate-500 mb-2">Original Proof Input Hash</div>
        <code className="text-sm font-mono text-slate-700 dark:text-slate-300 break-all">
          {truncateHash(proofInputHash, 16)}
        </code>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-700 dark:text-blue-300">
          <strong>Why this matters:</strong> Any modification to the inputs after proof generation
          causes the hash to change. The on-chain verifier will reject proofs with mismatched hashes,
          preventing tampering or replay attacks.
        </p>
      </div>
    </div>
  );
}
