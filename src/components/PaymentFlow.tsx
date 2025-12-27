'use client';

import { Check, Loader2, CreditCard, Shield, FileCheck, Zap } from 'lucide-react';

export type PaymentStep = 'policy' | 'proof' | 'submit' | 'payment';
export type StepStatus = 'pending' | 'active' | 'complete' | 'error';

interface PaymentFlowProps {
  currentStep: PaymentStep;
  stepStatuses: Record<PaymentStep, StepStatus>;
  onStepClick?: (step: PaymentStep) => void;
}

const STEPS: { id: PaymentStep; label: string; icon: typeof Shield }[] = [
  { id: 'policy', label: 'Policy Check', icon: Shield },
  { id: 'proof', label: 'Generate Proof', icon: FileCheck },
  { id: 'submit', label: 'Submit to Arc', icon: Zap },
  { id: 'payment', label: 'Execute Payment', icon: CreditCard },
];

export function PaymentFlow({ currentStep, stepStatuses, onStepClick }: PaymentFlowProps) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
        End-to-End Payment Flow
      </h3>

      <div className="flex items-center justify-between">
        {STEPS.map((step, index) => {
          const status = stepStatuses[step.id];
          const Icon = step.icon;
          const isActive = step.id === currentStep;

          return (
            <div key={step.id} className="flex items-center flex-1">
              {/* Step Circle */}
              <button
                onClick={() => onStepClick?.(step.id)}
                disabled={status === 'pending'}
                className={`relative flex flex-col items-center group ${
                  status === 'pending' ? 'cursor-not-allowed' : 'cursor-pointer'
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                    status === 'complete'
                      ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                      : status === 'active'
                      ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 ring-2 ring-purple-600 ring-offset-2 dark:ring-offset-slate-900'
                      : status === 'error'
                      ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
                  }`}
                >
                  {status === 'complete' ? (
                    <Check className="w-6 h-6" />
                  ) : status === 'active' ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <Icon className="w-6 h-6" />
                  )}
                </div>
                <span
                  className={`mt-2 text-xs font-medium transition-colors ${
                    status === 'complete'
                      ? 'text-green-600 dark:text-green-400'
                      : status === 'active'
                      ? 'text-purple-600 dark:text-purple-400'
                      : status === 'error'
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-slate-400'
                  }`}
                >
                  {step.label}
                </span>
              </button>

              {/* Connector Line */}
              {index < STEPS.length - 1 && (
                <div className="flex-1 h-0.5 mx-4">
                  <div
                    className={`h-full transition-colors ${
                      stepStatuses[STEPS[index + 1].id] !== 'pending'
                        ? 'bg-purple-600'
                        : 'bg-slate-200 dark:bg-slate-700'
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Current Step Description */}
      <div className="mt-8 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
        {currentStep === 'policy' && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Checking if the purchase complies with your spending policy...
          </p>
        )}
        {currentStep === 'proof' && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Generating a cryptographic proof that your agent followed the policy. This takes 4-12 seconds.
          </p>
        )}
        {currentStep === 'submit' && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Submitting the proof to Arc Testnet for on-chain verification...
          </p>
        )}
        {currentStep === 'payment' && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Executing the USDC payment with the verified proof attached.
          </p>
        )}
      </div>
    </div>
  );
}
