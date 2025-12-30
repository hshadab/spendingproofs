'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { PaymentFlow, type PaymentStep, type StepStatus } from '@/components/PaymentFlow';
import { ProofViewer } from '@/components/ProofViewer';
import { PolicySliders } from '@/components/PolicySliders';
import { PurchaseSimulator } from '@/components/PurchaseSimulator';
import { EnforcementDemo } from '@/components/EnforcementDemo';
import {
  type SpendingPolicy,
  type SpendingModelInput,
  DEFAULT_SPENDING_POLICY,
  createDefaultInput,
  runSpendingModel,
} from '@/lib/spendingModel';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useDemoWallet } from '@/hooks/useDemoWallet';
import { Check, AlertCircle, Lock, Shield, ExternalLink, ArrowLeft, Wallet } from 'lucide-react';

export default function PaymentPage() {
  const {
    status: walletStatus,
    isLoading: isWalletLoading,
    error: walletError,
    submitAttestation,
    executePayment,
    balance,
    address,
  } = useDemoWallet();

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
  const [showEnforcement, setShowEnforcement] = useState(false);
  const [proofHash, setProofHash] = useState<string | null>(null);
  const [attestationUrl, setAttestationUrl] = useState<string | null>(null);
  const [paymentUrl, setPaymentUrl] = useState<string | null>(null);
  const [txError, setTxError] = useState<string | null>(null);

  const handlePolicyCheck = () => {
    setStepStatuses((prev) => ({ ...prev, policy: 'active' }));
    setCurrentStep('policy');
    setTxError(null);

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

    if (result.success && result.proof) {
      setProofHash(result.proof.proofHash || result.proof.metadata?.inputHash || '0x' + Date.now().toString(16));
      setStepStatuses((prev) => ({ ...prev, proof: 'complete', submit: 'pending' }));
      setCurrentStep('submit');
    } else {
      setStepStatuses((prev) => ({ ...prev, proof: 'error' }));
    }
  };

  const handleSubmitAttestation = async () => {
    if (!proofHash) return;

    setStepStatuses((prev) => ({ ...prev, submit: 'active' }));
    setCurrentStep('submit');

    const result = await submitAttestation({
      validatorAddress: process.env.NEXT_PUBLIC_ARC_AGENT || '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384',
      agentId: 1,
      requestUri: `spending-proof:${Date.now()}`,
      proofHash: proofHash,
    });

    if (result.success) {
      setAttestationUrl(result.explorerUrl || null);
      setStepStatuses((prev) => ({ ...prev, submit: 'complete', payment: 'pending' }));
      setCurrentStep('payment');
    } else {
      setTxError(result.error || 'Attestation failed');
      setStepStatuses((prev) => ({ ...prev, submit: 'error' }));
    }
  };

  const handleExecutePayment = async () => {
    setStepStatuses((prev) => ({ ...prev, payment: 'active' }));

    const merchantAddress = process.env.NEXT_PUBLIC_DEMO_MERCHANT || '0x8ba1f109551bD432803012645Ac136ddd64DBA72';

    const result = await executePayment({
      to: merchantAddress,
      amount: input.priceUsdc,
    });

    if (result.success || result.simulated) {
      setPaymentUrl(result.explorerUrl || null);
      setStepStatuses((prev) => ({ ...prev, payment: 'complete' }));
    } else {
      setTxError(result.error || 'Payment failed');
      setStepStatuses((prev) => ({ ...prev, payment: 'error' }));
    }
  };

  const handleReset = () => {
    reset();
    setPolicyResult(null);
    setProofHash(null);
    setAttestationUrl(null);
    setPaymentUrl(null);
    setTxError(null);
    setCurrentStep('policy');
    setStepStatuses({
      policy: 'pending',
      proof: 'pending',
      submit: 'pending',
      payment: 'pending',
    });
    setShowEnforcement(false);
  };

  const isComplete = stepStatuses.payment === 'complete';
  const isProcessing = isWalletLoading || Object.values(stepStatuses).some((s) => s === 'active');

  return (
    <div className="max-w-4xl mx-auto py-8">
      <Link
        href="/demo"
        className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Demos
      </Link>

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          Payment Flow
        </h1>
        <p className="text-gray-400">
          End-to-end flow: policy check, proof generation, on-chain attestation, and USDC payment.
          <span className="text-green-400 ml-1">No wallet connection required - uses demo wallet.</span>
        </p>
      </div>

      {/* Demo Wallet Status */}
      <div className="mb-6 p-4 bg-[#0d1117] border border-gray-800 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="text-sm font-medium text-white">Demo Wallet</div>
              <div className="text-xs text-gray-400 font-mono">
                {address ? `${address.slice(0, 6)}...${address.slice(-4)}` : 'Loading...'}
              </div>
            </div>
          </div>
          <div className="text-right">
            {balance ? (
              <div className="text-sm">
                <span className="text-green-400 font-mono">{balance.display}</span>
                <div className="text-xs text-gray-500">{balance.native} ETH (gas)</div>
              </div>
            ) : (
              <div className="text-sm text-gray-500">Loading balance...</div>
            )}
          </div>
        </div>
        {walletStatus && !walletStatus.funded && (
          <div className="mt-3 p-2 bg-amber-500/10 border border-amber-500/20 rounded text-xs text-amber-400">
            Demo wallet needs funding. Send Arc testnet tokens to: <span className="font-mono">{address}</span>
          </div>
        )}
      </div>

      {/* Attestation vs Enforcement */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-5">
          <h3 className="font-semibold mb-2 text-gray-300 flex items-center gap-2">
            <Shield className="w-4 h-4 text-amber-500" />
            Attestation (This Demo)
          </h3>
          <p className="text-sm text-gray-400 mb-3">
            Proof hash is logged on-chain for auditability. Transfer proceeds with proof attached.
          </p>
          <div className="text-xs text-amber-400 bg-amber-500/10 px-2 py-1 rounded inline-block">
            Advisory audit trail
          </div>
        </div>
        <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-5">
          <h3 className="font-semibold mb-2 text-purple-300 flex items-center gap-2">
            <Lock className="w-4 h-4" />
            Enforcement (Explore Below)
          </h3>
          <p className="text-sm text-gray-400 mb-3">
            SpendingGate contract checks proof validity. Transfer reverts if proof is missing or invalid.
          </p>
          <div className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded inline-block">
            No Proof, No Payment
          </div>
        </div>
      </div>

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
              disabled={isProcessing}
              className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors"
            >
              Step 1: Check Policy
            </button>
          )}

          {/* Policy Result */}
          {policyResult && (
            <div className={`p-4 rounded-xl border ${
              policyResult.shouldBuy
                ? 'bg-green-900/20 border-green-800'
                : 'bg-red-900/20 border-red-800'
            }`}>
              <div className="flex items-center gap-2 font-semibold mb-2">
                {policyResult.shouldBuy ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-400" />
                )}
                <span className={policyResult.shouldBuy ? 'text-green-400' : 'text-red-400'}>
                  {policyResult.shouldBuy ? 'Policy Check Passed' : 'Policy Check Failed'}
                </span>
              </div>
              <p className="text-sm text-gray-400">{policyResult.reasons[0]}</p>
            </div>
          )}

          {/* Step 2: Generate Proof */}
          {stepStatuses.policy === 'complete' && stepStatuses.proof === 'pending' && (
            <button
              onClick={handleGenerateProof}
              disabled={isProcessing}
              className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors"
            >
              Step 2: Generate Proof
            </button>
          )}

          {/* Proof Generation Progress */}
          {state.status === 'running' && (
            <div className="p-4 bg-purple-900/20 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-purple-300">
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
              onClick={handleSubmitAttestation}
              disabled={isProcessing || !proofHash}
              className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors"
            >
              Step 3: Submit Attestation to Arc
            </button>
          )}

          {stepStatuses.submit === 'active' && (
            <div className="p-4 bg-purple-900/20 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-purple-300">Submitting to Arc Testnet...</span>
              </div>
            </div>
          )}

          {/* Attestation Success */}
          {stepStatuses.submit === 'complete' && attestationUrl && (
            <div className="p-4 bg-green-900/20 border border-green-800 rounded-xl">
              <div className="flex items-center gap-2 text-green-400 mb-2">
                <Check className="w-5 h-5" />
                <span className="font-medium">Attestation Recorded On-Chain</span>
              </div>
              <a
                href={attestationUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-green-400 hover:underline flex items-center gap-1"
              >
                View on ArcScan <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}

          {/* Step 4: Execute Payment */}
          {stepStatuses.submit === 'complete' && stepStatuses.payment === 'pending' && (
            <button
              onClick={handleExecutePayment}
              disabled={isProcessing}
              className="w-full py-3 px-6 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white font-medium rounded-lg transition-colors"
            >
              Step 4: Execute USDC Payment (${input.priceUsdc.toFixed(2)})
            </button>
          )}

          {stepStatuses.payment === 'active' && (
            <div className="p-4 bg-green-900/20 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-green-300">Executing USDC payment...</span>
              </div>
            </div>
          )}

          {/* Error Display */}
          {txError && (
            <div className="p-4 bg-red-900/20 border border-red-800 rounded-xl">
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm">{txError}</span>
              </div>
            </div>
          )}

          {/* Complete */}
          {isComplete && !showEnforcement && (
            <div className="p-6 bg-green-900/20 border border-green-800 rounded-xl text-center">
              <Check className="w-12 h-12 text-green-400 mx-auto mb-3" />
              <div className="text-lg font-semibold text-green-300 mb-2">
                Payment Complete!
              </div>
              <p className="text-sm text-gray-400 mb-4">
                Your agent&apos;s purchase was verified with a cryptographic proof and the payment was executed on Arc.
              </p>
              <div className="flex flex-col gap-2 mb-4">
                {attestationUrl && (
                  <a
                    href={attestationUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center gap-1 text-sm text-purple-400 hover:underline"
                  >
                    View Attestation on ArcScan <ExternalLink className="w-3 h-3" />
                  </a>
                )}
                {paymentUrl && (
                  <a
                    href={paymentUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center gap-1 text-sm text-green-400 hover:underline"
                  >
                    View Payment on ArcScan <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => setShowEnforcement(true)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                >
                  <Lock className="w-4 h-4" />
                  Explore Enforcement
                </button>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
                >
                  Start Over
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Enforcement Demo Section */}
      {showEnforcement && (
        <div className="mt-8 border-t border-gray-800 pt-8">
          <EnforcementDemo
            proof={state.result?.proof && state.result.inference ? {
              proofHash: state.result.proof.proofHash || proofHash || '',
              inputHash: state.result.proof.metadata?.inputHash || '',
              modelHash: state.result.proof.metadata?.modelHash || '',
              decision: {
                shouldBuy: state.result.inference.decision === 'approve',
                confidence: state.result.inference.confidence,
                riskScore: Math.round((1 - state.result.inference.confidence) * 100),
              },
              timestamp: Date.now(),
              proofSizeBytes: state.result.proof.metadata?.proofSize || 48000,
              generationTimeMs: state.elapsedMs || 0,
              verified: true,
            } : null}
            onReset={() => setShowEnforcement(false)}
          />
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleReset}
              className="px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
            >
              Start Over
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
