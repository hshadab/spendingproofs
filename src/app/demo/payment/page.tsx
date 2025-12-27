'use client';

import { useState } from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount } from 'wagmi';
import { PaymentFlow, type PaymentStep, type StepStatus } from '@/components/PaymentFlow';
import { ProofViewer } from '@/components/ProofViewer';
import { PolicySliders } from '@/components/PolicySliders';
import { PurchaseSimulator } from '@/components/PurchaseSimulator';
import {
  type SpendingPolicy,
  type SpendingModelInput,
  DEFAULT_SPENDING_POLICY,
  createDefaultInput,
  runSpendingModel,
} from '@/lib/spendingModel';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { Check, AlertCircle } from 'lucide-react';

export default function PaymentPage() {
  const { isConnected, address } = useAccount();
  const [policy, setPolicy] = useState<SpendingPolicy>(DEFAULT_SPENDING_POLICY);
  const [input, setInput] = useState<SpendingModelInput>(createDefaultInput());
  const { state, generateProof, reset } = useProofGeneration();

  const [currentStep, setCurrentStep] = useState<PaymentStep>('policy');
  const [stepStatuses, setStepStatuses] = useState<Record<PaymentStep, StepStatus>>({
    policy: 'pending',
    proof: 'pending',
    submit: 'pending',
    payment: 'pending',
  });
  const [policyResult, setPolicyResult] = useState<ReturnType<typeof runSpendingModel> | null>(null);

  const handlePolicyCheck = () => {
    setStepStatuses((prev) => ({ ...prev, policy: 'active' }));
    setCurrentStep('policy');

    // Run policy check
    const result = runSpendingModel(input, policy);
    setPolicyResult(result);

    setTimeout(() => {
      setStepStatuses((prev) => ({
        ...prev,
        policy: result.shouldBuy ? 'complete' : 'error',
      }));

      if (result.shouldBuy) {
        setCurrentStep('proof');
        setStepStatuses((prev) => ({ ...prev, proof: 'pending' }));
      }
    }, 1000);
  };

  const handleGenerateProof = async () => {
    setStepStatuses((prev) => ({ ...prev, proof: 'active' }));
    setCurrentStep('proof');

    const result = await generateProof(input);

    if (result.success) {
      setStepStatuses((prev) => ({ ...prev, proof: 'complete', submit: 'pending' }));
      setCurrentStep('submit');
    } else {
      setStepStatuses((prev) => ({ ...prev, proof: 'error' }));
    }
  };

  const handleSubmitToArc = async () => {
    setStepStatuses((prev) => ({ ...prev, submit: 'active' }));
    setCurrentStep('submit');

    // Simulate on-chain submission
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setStepStatuses((prev) => ({ ...prev, submit: 'complete', payment: 'pending' }));
    setCurrentStep('payment');
  };

  const handleExecutePayment = async () => {
    setStepStatuses((prev) => ({ ...prev, payment: 'active' }));

    // Simulate payment
    await new Promise((resolve) => setTimeout(resolve, 1500));

    setStepStatuses((prev) => ({ ...prev, payment: 'complete' }));
  };

  const handleReset = () => {
    reset();
    setPolicyResult(null);
    setCurrentStep('policy');
    setStepStatuses({
      policy: 'pending',
      proof: 'pending',
      submit: 'pending',
      payment: 'pending',
    });
  };

  const isComplete = stepStatuses.payment === 'complete';
  const isProcessing = Object.values(stepStatuses).some((s) => s === 'active');

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          End-to-End Payment
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Complete flow: policy check, proof generation, on-chain verification, and USDC payment.
        </p>
      </div>

      {/* Wallet Connection */}
      <div className="mb-6 flex justify-end">
        <ConnectButton />
      </div>

      {!isConnected ? (
        <div className="text-center py-12 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            Connect your wallet to start the payment flow
          </p>
          <ConnectButton />
        </div>
      ) : (
        <>
          {/* Payment Flow Visualization */}
          <div className="mb-8">
            <PaymentFlow
              currentStep={currentStep}
              stepStatuses={stepStatuses}
            />
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Left: Configuration */}
            <div className="space-y-6">
              <PolicySliders
                policy={policy}
                onChange={setPolicy}
                disabled={isProcessing || isComplete}
              />
              <PurchaseSimulator
                input={input}
                onChange={setInput}
                disabled={isProcessing || isComplete}
              />
            </div>

            {/* Right: Actions & Results */}
            <div className="space-y-6">
              {/* Step 1: Policy Check */}
              {stepStatuses.policy === 'pending' && (
                <button
                  onClick={handlePolicyCheck}
                  className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors"
                >
                  Step 1: Check Policy
                </button>
              )}

              {/* Policy Result */}
              {policyResult && (
                <div className={`p-4 rounded-xl border ${
                  policyResult.shouldBuy
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                    : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                }`}>
                  <div className="flex items-center gap-2 font-semibold mb-2">
                    {policyResult.shouldBuy ? (
                      <Check className="w-5 h-5 text-green-600" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-600" />
                    )}
                    {policyResult.shouldBuy ? 'Policy Check Passed' : 'Policy Check Failed'}
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    {policyResult.reasons[0]}
                  </p>
                </div>
              )}

              {/* Step 2: Generate Proof */}
              {stepStatuses.policy === 'complete' && stepStatuses.proof === 'pending' && (
                <button
                  onClick={handleGenerateProof}
                  className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors"
                >
                  Step 2: Generate Proof
                </button>
              )}

              {/* Proof Generation Progress */}
              {state.status === 'running' && (
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin" />
                    <span className="text-purple-700 dark:text-purple-300">
                      Generating SNARK proof... ({(state.elapsedMs / 1000).toFixed(1)}s)
                    </span>
                  </div>
                </div>
              )}

              {/* Proof Result */}
              {state.result?.proof && (
                <ProofViewer
                  proof={state.result.proof}
                  inference={state.result.inference}
                />
              )}

              {/* Step 3: Submit to Arc */}
              {stepStatuses.proof === 'complete' && stepStatuses.submit === 'pending' && (
                <button
                  onClick={handleSubmitToArc}
                  className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors"
                >
                  Step 3: Submit to Arc
                </button>
              )}

              {stepStatuses.submit === 'active' && (
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin" />
                    <span className="text-purple-700 dark:text-purple-300">
                      Submitting proof to Arc Testnet...
                    </span>
                  </div>
                </div>
              )}

              {/* Step 4: Execute Payment */}
              {stepStatuses.submit === 'complete' && stepStatuses.payment === 'pending' && (
                <button
                  onClick={handleExecutePayment}
                  className="w-full py-3 px-6 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
                >
                  Step 4: Execute USDC Payment
                </button>
              )}

              {stepStatuses.payment === 'active' && (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
                    <span className="text-green-700 dark:text-green-300">
                      Executing USDC payment...
                    </span>
                  </div>
                </div>
              )}

              {/* Complete */}
              {isComplete && (
                <div className="p-6 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl text-center">
                  <Check className="w-12 h-12 text-green-600 mx-auto mb-3" />
                  <div className="text-lg font-semibold text-green-700 dark:text-green-300 mb-2">
                    Payment Complete!
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                    Your agent's purchase was verified with a cryptographic proof and the payment was executed on Arc.
                  </p>
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg text-sm font-medium transition-colors"
                  >
                    Start Over
                  </button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
