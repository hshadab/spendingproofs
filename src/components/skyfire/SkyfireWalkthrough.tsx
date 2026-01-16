'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  User,
  Zap,
  CreditCard,
  CheckCircle2,
  Shield,
  ExternalLink,
  FileCheck,
  Info,
  Key,
  ArrowRight,
  Package,
  Lock,
  Globe,
} from 'lucide-react';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useSkyfireAgent } from '@/hooks/useSkyfireAgent';
import { useSkyfirePayment } from '@/hooks/useSkyfirePayment';
import { ProofProgress } from '@/components/ProofProgress';
import {
  createDefaultInput,
  runSpendingModel,
  DEFAULT_SPENDING_POLICY,
  type SpendingModelInput,
} from '@/lib/spendingModel';
import { getExplorerTxUrl } from '@/lib/config';
import type { SpendingProof } from '@/lib/types';

// Demo wallet address
const DEMO_RECIPIENT = '0x742d35Cc6634C0532925a3b844Bc9e7595f1E5b8' as const;

// Phase definitions
const PHASES = ['intro', 'identity', 'proof', 'attestation', 'payment', 'complete'] as const;
type Phase = typeof PHASES[number];

// Phase colors
const PHASE_COLORS: Record<Phase, string> = {
  intro: 'from-orange-500 to-amber-500',
  identity: 'from-purple-500 to-pink-500',
  proof: 'from-yellow-500 to-orange-500',
  attestation: 'from-blue-500 to-cyan-500',
  payment: 'from-green-500 to-emerald-500',
  complete: 'from-green-400 to-cyan-400',
};

const PHASE_LABELS: Record<Phase, string> = {
  intro: 'Introduction',
  identity: 'Skyfire KYA',
  proof: 'zkML Spending Policy Proof',
  attestation: 'Attestation',
  payment: 'Payment',
  complete: 'Complete',
};

// Business annotation
interface BusinessAnnotation {
  title: string;
  takeaway: string;
  color: 'skyfire' | 'zkml' | 'arc' | 'combined';
  metric?: string;
  metricLabel?: string;
}

// Walkthrough step
interface WalkthroughStep {
  id: string;
  phase: Phase;
  title: string;
  description: string;
  skyfireNote?: string;
  docUrl?: string;
  docLabel?: string;
  duration: number;
  annotation?: BusinessAnnotation;
}

// Annotation colors
const ANNOTATION_COLORS = {
  skyfire: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/50',
    title: 'text-orange-400',
  },
  zkml: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/50',
    title: 'text-yellow-400',
  },
  arc: {
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/50',
    title: 'text-purple-400',
  },
  combined: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/50',
    title: 'text-green-400',
  },
};

// Walkthrough steps - positive framing showing how zkML extends Skyfire
// Annotations only appear at the END of each phase as a brief takeaway
const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'Extending Skyfire with zkML',
    description: 'Skyfire provides powerful KYA (Know Your Agent) identity verification and payment infrastructure for AI agents. This demo shows how zkML spending policy proofs can extend Skyfire\'s capabilities — adding cryptographic verification of agent decision-making.',
    skyfireNote: 'Skyfire KYA establishes verified agent identity. zkML adds verifiable policy compliance.',
    docUrl: 'https://docs.skyfire.xyz',
    docLabel: 'Skyfire Docs',
    duration: 5000,
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'Complementary Capabilities',
    description: 'Skyfire\'s KYA tokens answer "Who is this agent?" with cryptographically signed credentials. zkML extends this by answering "Did the agent follow its spending policy?" — useful for enterprises requiring auditable policy compliance.',
    skyfireNote: 'Skyfire handles identity and payments. zkML adds an optional policy verification layer.',
    docUrl: 'https://docs.skyfire.xyz/kya-pay-tokens',
    docLabel: 'KYA+Pay Tokens',
    duration: 5000,
    annotation: {
      title: 'Skyfire + zkML',
      takeaway: 'Skyfire: verified identity & payments. zkML: verified spending policy compliance.',
      color: 'combined',
      metric: '2',
      metricLabel: 'verification layers',
    },
  },
  {
    id: 'identity-1',
    phase: 'identity',
    title: 'Skyfire Agent Identity',
    description: 'Skyfire\'s KYA (Know Your Agent) system provides verified agent identities. Each agent gets cryptographically signed credentials that prove who they are to any counterparty.',
    skyfireNote: 'Real Skyfire API: Creating agent identity with KYA credentials.',
    docUrl: 'https://docs.skyfire.xyz/buyers/buyer-onboarding',
    docLabel: 'Buyer Onboarding',
    duration: 6000,
  },
  {
    id: 'identity-2',
    phase: 'identity',
    title: 'KYA Token Issued',
    description: 'Skyfire issues a signed JWT token containing the agent\'s verified identity. This token can be presented to any seller to prove the agent is authorized to transact.',
    skyfireNote: 'Real Skyfire JWT with ES256 signature from api-sandbox.skyfire.xyz',
    duration: 4000,
    annotation: {
      title: 'KYA Identity Complete',
      takeaway: 'Skyfire KYA provides the verified identity foundation for agent commerce.',
      color: 'skyfire',
      metric: 'KYA',
      metricLabel: 'verified',
    },
  },
  {
    id: 'proof-1',
    phase: 'proof',
    title: 'Spending Policy Evaluation',
    description: 'For enterprises requiring policy compliance, zkML can prove the agent evaluated its spending policy before making a purchase. The policy checks vendor risk, budget limits, and compliance requirements.',
    skyfireNote: 'This extends Skyfire with optional policy verification for regulated use cases.',
    duration: 6000,
  },
  {
    id: 'proof-2',
    phase: 'proof',
    title: 'Generating zkML Spending Policy Proof',
    description: 'JOLT-Atlas runs the spending policy model inside a SNARK circuit, generating a ~48KB cryptographic proof. This proves the exact model ran on the exact inputs.',
    skyfireNote: 'Proof generation via JOLT-Atlas prover. Adds policy compliance to Skyfire identity.',
    duration: 8000,
  },
  {
    id: 'proof-3',
    phase: 'proof',
    title: 'Proof Verified',
    description: 'The proof is verified in under 150ms. Combined with Skyfire\'s KYA token, we now have both verified identity AND verified policy compliance.',
    skyfireNote: 'Skyfire identity + zkML spending policy proof = complete verification for enterprise use cases.',
    duration: 4000,
    annotation: {
      title: 'zkML Spending Policy Proof Complete',
      takeaway: 'Skyfire KYA (who) + zkML proof (policy compliance) = enhanced verification.',
      color: 'zkml',
      metric: '~48KB',
      metricLabel: 'SNARK proof',
    },
  },
  {
    id: 'attestation-1',
    phase: 'attestation',
    title: 'Binding Proof to Identity',
    description: 'We create a verificationHash that binds the zkML proof to the Skyfire agent identity. This links the policy compliance proof to the verified agent.',
    skyfireNote: 'verificationHash = hash(proofHash + Skyfire agentId + decision + confidence)',
    duration: 5000,
  },
  {
    id: 'attestation-2',
    phase: 'attestation',
    title: 'On-Chain Attestation',
    description: 'The verificationHash is recorded on Arc testnet, creating an immutable audit trail. This optional step provides permanent proof for compliance requirements.',
    skyfireNote: 'Adds on-chain auditability to Skyfire\'s payment infrastructure.',
    duration: 6000,
    annotation: {
      title: 'Attestation Complete',
      takeaway: 'On-chain record provides permanent, auditable proof of policy compliance.',
      color: 'arc',
      metric: 'Arc',
      metricLabel: 'testnet',
    },
  },
  {
    id: 'payment-1',
    phase: 'payment',
    title: 'Skyfire PAY Token',
    description: 'Skyfire generates a KYA+PAY token that authorizes the payment. The buyerTag includes a reference to the zkML proof for traceability.',
    skyfireNote: 'Real Skyfire KYA+PAY JWT with proof binding in buyerTag metadata.',
    docUrl: 'https://docs.skyfire.xyz/reference/create-token',
    docLabel: 'Create Token API',
    duration: 4000,
  },
  {
    id: 'payment-2',
    phase: 'payment',
    title: 'Payment Executed',
    description: 'The SpendingGateWallet verifies the on-chain attestation before releasing funds. This demonstrates how zkML can add policy enforcement to Skyfire payments.',
    skyfireNote: 'Skyfire handles identity + payment. Smart contract adds optional policy enforcement.',
    duration: 5000,
    annotation: {
      title: 'Payment Complete',
      takeaway: 'Skyfire payments enhanced with zkML spending policy verification.',
      color: 'arc',
      metric: '$0.01',
      metricLabel: 'USDC transferred',
    },
  },
  {
    id: 'complete-1',
    phase: 'complete',
    title: 'Extended Agent Commerce',
    description: 'Complete flow: Skyfire KYA identity → zkML spending policy proof → on-chain attestation → Skyfire payment. This demonstrates how zkML can extend Skyfire\'s powerful agent commerce infrastructure.',
    duration: 8000,
    annotation: {
      title: 'Skyfire Extended',
      takeaway: 'Skyfire provides the foundation. zkML adds optional spending policy verification for enterprise needs.',
      color: 'combined',
      metric: '4',
      metricLabel: 'verification steps',
    },
  },
];

export function SkyfireWalkthrough() {
  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);
  const processedStepRef = useRef<string | null>(null);

  // Demo state
  const [spendingProof, setSpendingProof] = useState<SpendingProof | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [attestationTxHash, setAttestationTxHash] = useState<string | null>(null);
  const [verificationHash, setVerificationHash] = useState<string | null>(null);
  const [agentThoughts, setAgentThoughts] = useState<string[]>([]);

  // Annotation overlay
  const [showingAnnotation, setShowingAnnotation] = useState(false);
  const [annotationData, setAnnotationData] = useState<{ annotation: BusinessAnnotation; stepTitle: string } | null>(null);
  const annotationTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Hooks
  const { state: proofState, generateProof, reset: resetProof } = useProofGeneration();
  const [input] = useState<SpendingModelInput>(createDefaultInput());
  const {
    agent,
    kyaToken,
    isCreating,
    createAgent,
    clearAgent,
  } = useSkyfireAgent();
  const {
    transferResult,
    isTransferring,
    executeVerifiedTransfer,
    reset: resetPayment,
  } = useSkyfirePayment();

  const currentStep = WALKTHROUGH_STEPS[currentStepIndex];
  const currentPhaseIndex = PHASES.indexOf(currentStep.phase);

  // Keep ref in sync
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Update state when transfer completes
  useEffect(() => {
    if (transferResult?.success && transferResult.transfer) {
      setTxHash(transferResult.transfer.txHash);
      setAttestationTxHash(transferResult.transfer.attestation.txHash || null);
      setVerificationHash(transferResult.transfer.zkml.verificationHash || null);
    }
  }, [transferResult]);

  // Main step effect
  useEffect(() => {
    const step = WALKTHROUGH_STEPS[currentStepIndex];
    const stepKey = step.id;
    const alreadyProcessed = processedStepRef.current === stepKey;

    // Clear any existing timers when step changes or play state changes
    if (timerRef.current) clearTimeout(timerRef.current);
    if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);

    if (!alreadyProcessed) {
      processedStepRef.current = stepKey;

      // Phase-specific effects
      if (step.phase === 'identity' && step.id === 'identity-1') {
        setAgentThoughts(['Connecting to Skyfire API...']);
        createAgent('zkML Demo Agent').then(newAgent => {
          setAgentThoughts(prev => [...prev, `✓ Agent ID: ${newAgent.id}`]);
          setAgentThoughts(prev => [...prev, `✓ Issuer: ${newAgent.kyaCredentials.issuer}`]);
          setAgentThoughts(prev => [...prev, '✓ KYA credentials verified']);
        }).catch(() => {
          setAgentThoughts(prev => [...prev, '✓ Demo mode agent created']);
        });
      }

      if (step.phase === 'identity' && step.id === 'identity-2' && agent) {
        setTimeout(() => {
          setAgentThoughts(prev => [...prev, '✓ KYA token generated (ES256 JWT)']);
          if (kyaToken) {
            setAgentThoughts(prev => [...prev, `✓ Token expires: ${new Date(kyaToken.expiresAt).toLocaleTimeString()}`]);
          }
        }, 1000);
      }

      if (step.phase === 'proof' && step.id === 'proof-1') {
        setAgentThoughts(prev => [...prev, 'Evaluating spending policy...']);
        generateProof(input).then(result => {
          if (result.success && result.proof) {
            const decision = runSpendingModel(input, DEFAULT_SPENDING_POLICY);
            const proof: SpendingProof = {
              proof: result.proof.proof,
              proofHash: result.proof.proofHash,
              inputHash: result.proof.metadata.inputHash,
              modelHash: result.proof.metadata.modelHash,
              programIo: result.proof.programIo,
              decision: {
                shouldBuy: decision.shouldBuy,
                confidence: decision.confidence / 100,
                riskScore: decision.riskScore / 100,
              },
              timestamp: result.proof.timestamp,
              proofSizeBytes: result.proof.metadata.proofSize,
              generationTimeMs: result.generationTimeMs,
              verified: true,
              txIntentHash: result.proof.metadata.txIntentHash,
            };
            setSpendingProof(proof);
            setAgentThoughts(prev => [...prev, `✓ Decision: ${decision.shouldBuy ? 'APPROVE' : 'DENY'}`]);
            setAgentThoughts(prev => [...prev, `✓ Confidence: ${decision.confidence}%`]);
          }
        });
      }

      if (step.phase === 'payment' && step.id === 'payment-1' && spendingProof && agent) {
        setAgentThoughts(prev => [...prev, 'Generating Skyfire PAY token...']);
        // Execute verified transfer (includes attestation)
        executeVerifiedTransfer(
          DEMO_RECIPIENT,
          0.01,
          spendingProof.proofHash,
          {
            decision: spendingProof.decision.shouldBuy,
            confidence: spendingProof.decision.confidence,
          },
          agent.id,
          agent.name
        ).then(result => {
          if (result.success) {
            setAgentThoughts(prev => [...prev, '✓ Payment executed successfully']);
          }
        });
      }
    }

    // Timer for advancing - only run if playing
    // Annotations appear at the END of the step as a brief takeaway (2.5s)
    const stepAnnotation = step.annotation;
    if (isPlaying) {
      timerRef.current = setTimeout(() => {
        if (isPlayingRef.current) {
          if (stepAnnotation) {
            // Show annotation immediately at end of step
            setAnnotationData({ annotation: stepAnnotation, stepTitle: step.title });
            setShowingAnnotation(true);

            // Brief display (2.5s) then advance
            annotationTimerRef.current = setTimeout(() => {
              setShowingAnnotation(false);
              setAnnotationData(null);
              advanceStep();
            }, 2500);
          } else {
            advanceStep();
          }
        }
      }, step.duration);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
    };
  }, [currentStepIndex, isPlaying, generateProof, input, agent, spendingProof, createAgent, executeVerifiedTransfer, kyaToken]);

  const advanceStep = () => {
    setCurrentStepIndex(prev => {
      const next = prev + 1;
      if (next >= WALKTHROUGH_STEPS.length) {
        setIsPlaying(false);
        return prev;
      }
      return next;
    });
  };

  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleNext = useCallback(() => {
    setShowingAnnotation(false);
    setAnnotationData(null);
    setCurrentStepIndex(prev => Math.min(prev + 1, WALKTHROUGH_STEPS.length - 1));
  }, []);

  const handlePrevious = useCallback(() => {
    setShowingAnnotation(false);
    setAnnotationData(null);
    setCurrentStepIndex(prev => Math.max(prev - 1, 0));
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentStepIndex(0);
    setSpendingProof(null);
    setTxHash(null);
    setAttestationTxHash(null);
    setVerificationHash(null);
    setAgentThoughts([]);
    setShowingAnnotation(false);
    setAnnotationData(null);
    processedStepRef.current = null;
    resetProof();
    resetPayment();
    clearAgent();
    if (timerRef.current) clearTimeout(timerRef.current);
    if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
  }, [resetProof, resetPayment, clearAgent]);

  const getPhaseIcon = (phase: Phase) => {
    switch (phase) {
      case 'intro': return <Info className="w-4 h-4" />;
      case 'identity': return <User className="w-4 h-4" />;
      case 'proof': return <Zap className="w-4 h-4" />;
      case 'attestation': return <Shield className="w-4 h-4" />;
      case 'payment': return <CreditCard className="w-4 h-4" />;
      case 'complete': return <CheckCircle2 className="w-4 h-4" />;
    }
  };

  // Helper to decode JWT for display
  const decodeJwtPayload = (token: string): Record<string, unknown> | null => {
    try {
      const [, payload] = token.split('.');
      return JSON.parse(atob(payload));
    } catch {
      return null;
    }
  };

  return (
    <div className="flex flex-col">
      {/* Playback Controls */}
      <div className="mb-3 p-3 bg-[#0d1117] rounded-xl border border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              disabled={currentStepIndex <= 0}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50 border border-gray-700"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={handlePlayPause}
              className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg transition-all text-sm font-medium ${
                isPlaying
                  ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30'
                  : 'bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 border border-orange-500/30'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isPlaying ? 'Pause' : 'Play Demo'}</span>
            </button>
            <button
              onClick={handleNext}
              disabled={currentStepIndex >= WALKTHROUGH_STEPS.length - 1}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50 border border-gray-700"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            <div className="w-px h-6 bg-gray-700 mx-1" />
            <button
              onClick={handleReset}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors border border-gray-700"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://skyfire.xyz"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-orange-400 hover:text-orange-300"
            >
              <Globe className="w-3 h-3" />
              skyfire.xyz
            </a>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span>Step {currentStepIndex + 1}/{WALKTHROUGH_STEPS.length}</span>
              <div className="flex items-center gap-1">
                {PHASES.map((phase, i) => (
                  <div
                    key={phase}
                    className={`w-2 h-2 rounded-full transition-all ${
                      i < currentPhaseIndex
                        ? 'bg-green-500'
                        : i === currentPhaseIndex
                        ? `bg-gradient-to-r ${PHASE_COLORS[phase]}`
                        : 'bg-gray-700'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Annotation Overlay */}
      {showingAnnotation && annotationData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className={`max-w-md p-6 rounded-2xl border ${ANNOTATION_COLORS[annotationData.annotation.color].bg} ${ANNOTATION_COLORS[annotationData.annotation.color].border}`}>
            <div className="text-center">
              {annotationData.annotation.metric && (
                <div className={`text-5xl font-bold mb-2 ${ANNOTATION_COLORS[annotationData.annotation.color].title}`}>
                  {annotationData.annotation.metric}
                </div>
              )}
              {annotationData.annotation.metricLabel && (
                <div className="text-sm text-gray-400 mb-4">{annotationData.annotation.metricLabel}</div>
              )}
              <h3 className={`text-xl font-bold mb-2 ${ANNOTATION_COLORS[annotationData.annotation.color].title}`}>
                {annotationData.annotation.title}
              </h3>
              <p className="text-gray-300 text-sm">{annotationData.annotation.takeaway}</p>
              <button
                onClick={() => {
                  setShowingAnnotation(false);
                  setAnnotationData(null);
                  if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
                  advanceStep();
                }}
                className="mt-4 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded-lg transition-colors"
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Container */}
      <div className="flex min-h-[700px] bg-[#0a0a0a] rounded-2xl overflow-hidden border border-gray-800">
        {/* Left Sidebar */}
        <div className="w-80 bg-[#0d1117] border-r border-gray-800 flex flex-col flex-shrink-0">
          {/* Header */}
          <div className="p-4 border-b border-gray-800">
            {/* NovaNet Logo */}
            <img
              src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
              alt="NovaNet"
              className="h-6 mb-2"
            />
            <p className="text-xs text-gray-400 mt-1">
              Extending Skyfire with<br />
              <span className="text-orange-400 font-medium">Agent Spending Policy Verification</span>
            </p>
          </div>

          {/* Current Step */}
          <div className="p-4 flex-1 overflow-y-auto">
            {/* Phase Badge */}
            <div className="mb-2 flex items-center gap-2">
              <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${PHASE_COLORS[currentStep.phase]} text-white`}>
                {PHASE_LABELS[currentStep.phase]}
              </span>
              {currentStep.docUrl && (
                <a
                  href={currentStep.docUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-[10px] text-orange-400 hover:text-orange-300"
                >
                  {currentStep.docLabel || 'Docs'} <ExternalLink className="w-2.5 h-2.5" />
                </a>
              )}
            </div>

            {/* Title & Description */}
            <h3 className="text-lg font-bold text-white mb-2">{currentStep.title}</h3>
            <p className="text-sm text-gray-400 leading-relaxed mb-3">{currentStep.description}</p>

            {/* Skyfire Note */}
            {currentStep.skyfireNote && (
              <div className="p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg mb-3">
                <div className="flex items-start gap-2">
                  <Package className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-orange-300">{currentStep.skyfireNote}</p>
                </div>
              </div>
            )}

            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
                <span>Progress</span>
                <span>{currentPhaseIndex + 1}/{PHASES.length}</span>
              </div>
              <div className="flex items-center gap-1">
                {PHASES.map((phase, i) => (
                  <div
                    key={phase}
                    className={`h-1.5 flex-1 rounded-full transition-all ${
                      i < currentPhaseIndex
                        ? 'bg-green-500'
                        : i === currentPhaseIndex
                        ? `bg-gradient-to-r ${PHASE_COLORS[phase]} ${isPlaying ? 'animate-pulse' : ''}`
                        : 'bg-gray-700'
                    }`}
                  />
                ))}
              </div>
            </div>

            {/* Phase List */}
            <div className="space-y-1.5">
              {PHASES.map((phase, i) => {
                const isComplete = i < currentPhaseIndex;
                const isCurrent = i === currentPhaseIndex;
                return (
                  <div
                    key={phase}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded-md transition-all ${
                      isCurrent ? 'bg-gray-800' : ''
                    }`}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                      isComplete
                        ? 'bg-green-500 text-white'
                        : isCurrent
                        ? `bg-gradient-to-r ${PHASE_COLORS[phase]} text-white`
                        : 'bg-gray-700 text-gray-400'
                    }`}>
                      {isComplete ? <CheckCircle2 className="w-3 h-3" /> : getPhaseIcon(phase)}
                    </div>
                    <span className={`text-sm ${
                      isCurrent ? 'text-white font-medium' : isComplete ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {PHASE_LABELS[phase]}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-800">
            <div className="flex items-center justify-between text-[10px] text-gray-500">
              <a href="https://skyfire.xyz" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-orange-400">
                Powered by Skyfire <ExternalLink className="w-2.5 h-2.5" />
              </a>
              <span>Extended with zkML</span>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 p-6 bg-[#0a0a0a] overflow-y-auto">
          {/* Intro Phase */}
          {currentStep.phase === 'intro' && (
            <div>
              {/* Partner Logos */}
              <div className="flex items-center gap-4 mb-6">
                <a
                  href="https://skyfire.xyz"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-3 py-1.5 bg-orange-500/10 border border-orange-500/30 rounded-lg hover:bg-orange-500/20 transition-colors"
                >
                  <Zap className="w-4 h-4 text-orange-400" />
                  <span className="text-orange-400 font-medium text-sm">Skyfire KYA</span>
                  <ExternalLink className="w-3 h-3 text-orange-400/60" />
                </a>
                <span className="text-gray-600">+</span>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <Shield className="w-4 h-4 text-yellow-400" />
                  <span className="text-yellow-400 font-medium text-sm">JOLT-Atlas zkML</span>
                </div>
                <span className="text-gray-600">+</span>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <Lock className="w-4 h-4 text-purple-400" />
                  <span className="text-purple-400 font-medium text-sm">Arc Testnet</span>
                </div>
              </div>

              <h2 className="text-2xl font-bold mb-2 text-white">Extending Skyfire with zkML</h2>
              <p className="text-gray-400 max-w-2xl mb-6 text-sm">
                <span className="text-orange-400">Skyfire</span> provides powerful KYA (Know Your Agent) identity verification and payment infrastructure.
                <span className="text-yellow-400"> zkML</span> extends this with optional cryptographic proof of spending policy compliance — useful for enterprise and regulated use cases.
              </p>

              {/* What Each Provides */}
              <div className="grid grid-cols-2 gap-4 max-w-2xl mb-6">
                <div className="bg-[#0d1117] border border-orange-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Zap className="w-5 h-5 text-orange-400" />
                      <span className="text-orange-400 font-semibold">Skyfire Provides</span>
                    </div>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-orange-400" />
                      <span>KYA (Know Your Agent) Identity</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-orange-400" />
                      <span>Signed JWT Credentials</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-orange-400" />
                      <span>KYA+PAY Payment Tokens</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-orange-400" />
                      <span>Agent Commerce Infrastructure</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Shield className="w-5 h-5 text-yellow-400" />
                      <span className="text-yellow-400 font-semibold">zkML Adds</span>
                    </div>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>Spending Policy Verification</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>~48KB SNARK Proofs</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>On-Chain Attestations</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-300">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>Policy-Enforced Payments</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Use Cases */}
              {currentStep.id === 'intro-2' && (
                <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl">
                  <h3 className="text-sm font-semibold text-white mb-3">Enterprise Use Cases for zkML Extension</h3>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-2 bg-gray-800/50 rounded">
                      <span className="text-orange-400 font-medium">Financial Services</span>
                      <p className="text-gray-400 mt-1">Prove AI trading agents followed risk policies</p>
                    </div>
                    <div className="p-2 bg-gray-800/50 rounded">
                      <span className="text-orange-400 font-medium">Procurement</span>
                      <p className="text-gray-400 mt-1">Verify purchasing agents checked vendor compliance</p>
                    </div>
                    <div className="p-2 bg-gray-800/50 rounded">
                      <span className="text-orange-400 font-medium">Treasury</span>
                      <p className="text-gray-400 mt-1">Confirm agents stayed within budget allocations</p>
                    </div>
                    <div className="p-2 bg-gray-800/50 rounded">
                      <span className="text-orange-400 font-medium">Compliance</span>
                      <p className="text-gray-400 mt-1">Auditable proof of policy adherence for regulators</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Identity Phase */}
          {currentStep.phase === 'identity' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">Skyfire Agent Identity</h2>

              {/* Real API Indicator */}
              <div className="flex items-center gap-2 mb-4">
                <div className="flex items-center gap-1.5 px-2 py-1 bg-green-500/10 border border-green-500/30 rounded text-xs text-green-400">
                  <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                  Live Skyfire API
                </div>
                <span className="text-xs text-gray-500">api-sandbox.skyfire.xyz</span>
              </div>

              {/* Agent Thoughts / API Log */}
              <div className="bg-[#0d1117] border border-orange-500/30 rounded-xl p-4 mb-6 max-w-xl">
                <div className="flex items-center gap-2 mb-3">
                  <User className="w-5 h-5 text-orange-400" />
                  <span className="text-orange-400 font-semibold text-sm">Skyfire KYA Registration</span>
                </div>
                <div className="space-y-2 font-mono text-xs">
                  {agentThoughts.map((thought, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <ArrowRight className="w-3 h-3 text-orange-400 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-300">{thought}</span>
                    </div>
                  ))}
                  {isCreating && (
                    <div className="flex items-center gap-2 text-gray-500">
                      <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                      <span>Connecting to Skyfire...</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Agent Identity Card */}
              {agent && (
                <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4 max-w-xl">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-semibold text-sm">Skyfire Agent Identity</span>
                    <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded-full ml-2">
                      KYA Verified
                    </span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Agent ID</span>
                      <span className="text-white font-mono text-xs">{agent.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Name</span>
                      <span className="text-white font-medium">{agent.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Issuer</span>
                      <span className="text-orange-400 text-xs">{agent.kyaCredentials.issuer}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Status</span>
                      <span className="text-green-400 text-xs">{agent.kyaCredentials.verificationStatus}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* KYA Token Display - Show Real JWT */}
              {currentStep.id === 'identity-2' && kyaToken && (
                <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4 max-w-xl mt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Key className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-semibold text-sm">Skyfire KYA Token (Real JWT)</span>
                  </div>

                  {/* JWT Preview */}
                  <div className="bg-gray-900/50 p-2 rounded mb-3 overflow-hidden">
                    <code className="text-[10px] text-gray-400 break-all">
                      {kyaToken.token.substring(0, 80)}...
                    </code>
                  </div>

                  {/* Decoded Claims */}
                  {(() => {
                    const claims = decodeJwtPayload(kyaToken.token);
                    return claims && (
                      <div className="space-y-1.5 text-xs">
                        <div className="text-gray-500 mb-2">Decoded JWT Claims:</div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Issuer (iss)</span>
                          <span className="text-orange-400">{String(claims.iss)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Subject (sub)</span>
                          <span className="text-white font-mono">{String(claims.sub).substring(0, 12)}...</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Buyer Tag (btg)</span>
                          <span className="text-white font-mono">{String(claims.btg || 'N/A')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Environment</span>
                          <span className="text-green-400">{String(claims.env || 'sandbox')}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Expires</span>
                          <span className="text-white">{new Date(kyaToken.expiresAt).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          )}

          {/* Proof Phase */}
          {currentStep.phase === 'proof' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                {currentStep.id === 'proof-1' ? 'Spending Policy Inputs' :
                 currentStep.id === 'proof-2' ? 'Generating zkML Spending Policy Proof' :
                 'Local Proof Verification'}
              </h2>

              {/* Spending Policy Inputs - Detailed like ACK */}
              {currentStep.id === 'proof-1' && !spendingProof && (
                <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4 max-w-xl mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-semibold text-sm">ML Spending Policy</span>
                    <span className="px-2 py-0.5 bg-yellow-500/10 text-yellow-400 text-[10px] rounded ml-2">
                      Extends Skyfire
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">
                    The spending policy evaluates 6 factors to decide if a purchase should be approved — checking budget, vendor risk, and compliance together:
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Purchase Amount</span>
                      <span className="text-white font-mono">$0.01</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Available Budget</span>
                      <span className="text-white font-mono">$100</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Vendor Risk Score</span>
                      <span className="text-green-400 font-mono">15% <span className="text-gray-500 text-xs">(low risk)</span></span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Vendor Track Record</span>
                      <span className="text-green-400 font-mono">92% <span className="text-gray-500 text-xs">(excellent)</span></span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Category Budget Left</span>
                      <span className="text-white font-mono">$15</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Compliance Status</span>
                      <span className="text-green-400 font-mono">Verified</span>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-700 text-xs text-gray-500">
                    zkML proves this exact policy model ran on these exact inputs — unforgeable verification that extends Skyfire&apos;s identity layer.
                  </div>
                </div>
              )}

              {/* Proof Progress */}
              {(proofState.status === 'running' || proofState.status === 'error') && (
                <div className="max-w-xl mb-6">
                  <ProofProgress
                    status={proofState.status}
                    progress={proofState.progress}
                    elapsedMs={proofState.elapsedMs}
                    steps={proofState.steps}
                  />
                </div>
              )}

              {/* Proof Result */}
              {spendingProof && (
                <div className="space-y-4 max-w-xl">
                  <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                      <span className="text-green-400 font-semibold text-sm">zkML Proof Generated</span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Policy Decision</span>
                        <span className={`font-medium ${spendingProof.decision.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
                          {spendingProof.decision.shouldBuy ? 'APPROVE' : 'REJECT'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Confidence</span>
                        <span className="text-white">{(spendingProof.decision.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Proof Hash</span>
                        <span className="text-yellow-400 font-mono text-xs">
                          {spendingProof.proofHash.slice(0, 14)}...
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Proof Size</span>
                        <span className="text-white">{(spendingProof.proofSizeBytes / 1024).toFixed(1)} KB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Generation Time</span>
                        <span className="text-white">{(spendingProof.generationTimeMs / 1000).toFixed(1)}s</span>
                      </div>
                    </div>
                  </div>

                  {/* Link to Skyfire */}
                  {currentStep.id === 'proof-3' && (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Ready to Extend Skyfire Payment</span>
                      </div>
                      <p className="text-xs text-gray-400 mb-2">
                        zkML proof verified. Now binding to Skyfire agent identity for enhanced verification.
                      </p>
                      <div className="p-2 bg-orange-500/10 border border-orange-500/20 rounded text-xs text-orange-400">
                        Next: Create Skyfire PAY token with zkML proof binding
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Attestation Phase - Show both cards together */}
          {currentStep.phase === 'attestation' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                {currentStep.id === 'attestation-1' ? 'Binding Proof to Skyfire Identity' : 'On-Chain Attestation'}
              </h2>

              <div className="space-y-4 max-w-xl">
                {/* Card 1: Verification Hash Binding - Always show during attestation phase */}
                <div className="bg-[#0d1117] border border-blue-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <FileCheck className="w-5 h-5 text-blue-400" />
                    <span className="text-blue-400 font-semibold text-sm">Binding zkML to Skyfire</span>
                    {currentStep.id === 'attestation-2' && (
                      <CheckCircle2 className="w-4 h-4 text-green-400 ml-auto" />
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mb-3">
                    Creating a <span className="text-yellow-400">verificationHash</span> that binds the zkML proof
                    to the Skyfire agent identity:
                  </p>
                  <div className="bg-gray-900/50 p-3 rounded text-xs font-mono text-gray-300">
                    <div className="text-blue-400 mb-2">verificationHash = keccak256(</div>
                    <div className="pl-4 space-y-1">
                      <div><span className="text-yellow-400">proofHash</span>,      <span className="text-gray-500">// zkML proof</span></div>
                      <div><span className="text-orange-400">skyfireAgentId</span>, <span className="text-gray-500">// Skyfire KYA</span></div>
                      <div><span className="text-green-400">decision</span>,        <span className="text-gray-500">// shouldBuy</span></div>
                      <div><span className="text-purple-400">confidence</span>,     <span className="text-gray-500">// 0-100%</span></div>
                      <div><span className="text-cyan-400">timestamp</span>         <span className="text-gray-500">// when verified</span></div>
                    </div>
                    <div className="text-blue-400">)</div>
                  </div>
                </div>

                {/* Card 2: Arc Attestation - Show on both steps, with progress indicator on step 1 */}
                <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-semibold text-sm">Arc Testnet Attestation</span>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">
                    Recording verificationHash on-chain for permanent audit trail.
                  </p>
                  {currentStep.id === 'attestation-1' ? (
                    <div className="flex items-center gap-2 text-gray-500 text-sm">
                      <div className="w-2 h-2 bg-gray-500 rounded-full" />
                      <span>Pending verification hash...</span>
                    </div>
                  ) : !attestationTxHash ? (
                    <div className="flex items-center gap-2 text-purple-400 text-sm">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                      <span>Submitting attestation to Arc...</span>
                    </div>
                  ) : (
                    <div className="space-y-2 text-sm">
                      {verificationHash && (
                        <div className="flex justify-between">
                          <span className="text-gray-400">Verification Hash</span>
                          <span className="text-yellow-400 font-mono text-xs">
                            {verificationHash.slice(0, 14)}...
                          </span>
                        </div>
                      )}
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Attestation Tx</span>
                        <a
                          href={getExplorerTxUrl(attestationTxHash)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-cyan-400 font-mono text-xs flex items-center gap-1 hover:underline"
                        >
                          {attestationTxHash.slice(0, 10)}...{attestationTxHash.slice(-6)}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  )}
                </div>

                {/* Card 3: Completion status - only show when attestation is done */}
                {attestationTxHash && (
                  <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                      <span className="text-green-400 font-semibold text-sm">Attestation Complete</span>
                    </div>
                    <p className="text-xs text-gray-400">
                      Policy compliance now verifiable on-chain. Ready to extend Skyfire payment with zkML verification.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Payment Phase */}
          {currentStep.phase === 'payment' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                {currentStep.id === 'payment-1' ? 'Skyfire PAY Token' : 'Payment Executed'}
              </h2>

              {/* PAY Token Generation */}
              {currentStep.id === 'payment-1' && (
                <div className="space-y-4 max-w-xl">
                  {/* Real API Indicator */}
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-green-500/10 border border-green-500/30 rounded text-xs text-green-400">
                      <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                      Live Skyfire API
                    </div>
                    <span className="text-xs text-gray-500">Creating KYA+PAY token</span>
                  </div>

                  <div className="bg-[#0d1117] border border-orange-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Key className="w-5 h-5 text-orange-400" />
                      <span className="text-orange-400 font-semibold text-sm">Skyfire KYA+PAY Token</span>
                    </div>
                    <p className="text-xs text-gray-400 mb-3">
                      Real Skyfire payment token with zkML proof reference in buyerTag:
                    </p>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Token Type</span>
                        <span className="text-orange-400 font-medium">kya+pay</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Amount</span>
                        <span className="text-white font-medium">$0.01 USDC</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Buyer Tag</span>
                        <span className="text-yellow-400 font-mono text-xs">
                          {agent?.id}:proof:...
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Seller Service</span>
                        <span className="text-orange-400 text-xs">Skyfire Official</span>
                      </div>
                    </div>
                  </div>

                  {isTransferring && (
                    <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 text-yellow-400 text-sm">
                        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                        <span>Executing transfer with policy enforcement...</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Gated Transfer */}
              {currentStep.id === 'payment-2' && (
                <div className="space-y-4 max-w-xl">
                  {attestationTxHash && (
                    <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3">
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <CheckCircle2 className="w-4 h-4 text-green-400" />
                        <span>Policy compliance attested on-chain</span>
                        <span className="text-gray-600">•</span>
                        <span className="text-yellow-400 font-mono">{verificationHash?.slice(0, 10)}...</span>
                      </div>
                    </div>
                  )}

                  {!txHash ? (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <CreditCard className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Executing Payment</span>
                      </div>
                      <div className="flex items-center gap-2 text-gray-400 text-sm">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                        <span>SpendingGateWallet verifying attestation...</span>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Payment Complete</span>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Amount</span>
                          <span className="text-white font-medium">$0.01 USDC</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Method</span>
                          <span className="text-purple-400">SpendingGateWallet</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-400">Transfer Tx</span>
                          <a
                            href={getExplorerTxUrl(txHash)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-cyan-400 font-mono text-xs flex items-center gap-1 hover:underline"
                          >
                            {txHash.slice(0, 10)}...{txHash.slice(-6)}
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                      <div className="mt-3 p-2 bg-green-500/10 border border-green-500/20 rounded text-xs text-green-400">
                        Skyfire payment enhanced with zkML policy verification.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Complete Phase */}
          {currentStep.phase === 'complete' && (
            <div className="space-y-4 max-w-2xl">
              {/* Partner Badges */}
              <div className="flex items-center gap-3">
                <a
                  href="https://skyfire.xyz"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-2 py-1 bg-orange-500/10 border border-orange-500/30 rounded-lg hover:bg-orange-500/20"
                >
                  <Zap className="w-3 h-3 text-orange-400" />
                  <span className="text-orange-400 font-medium text-xs">Skyfire</span>
                </a>
                <span className="text-gray-600 text-xs">+</span>
                <div className="flex items-center gap-2 px-2 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <Shield className="w-3 h-3 text-yellow-400" />
                  <span className="text-yellow-400 font-medium text-xs">zkML</span>
                </div>
                <span className="text-gray-600 text-xs">+</span>
                <div className="flex items-center gap-2 px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <Lock className="w-3 h-3 text-purple-400" />
                  <span className="text-purple-400 font-medium text-xs">Arc</span>
                </div>
              </div>

              {/* Title */}
              <div>
                <h2 className="text-2xl font-bold text-white mb-2">Skyfire Extended with zkML</h2>
                <p className="text-gray-400 text-sm">
                  Demonstrated how zkML can extend Skyfire's agent commerce infrastructure with
                  optional spending policy verification for enterprise compliance needs.
                </p>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-4 gap-2">
                <div className="p-3 bg-[#0d1117] border border-orange-500/30 rounded-xl text-center">
                  <div className="text-xl font-bold text-orange-400">KYA</div>
                  <div className="text-[10px] text-gray-400">Skyfire Identity</div>
                </div>
                <div className="p-3 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                  <div className="text-xl font-bold text-yellow-400">~48KB</div>
                  <div className="text-[10px] text-gray-400">zkML Proof</div>
                </div>
                <div className="p-3 bg-[#0d1117] border border-purple-500/30 rounded-xl text-center">
                  <div className="text-xl font-bold text-purple-400">2</div>
                  <div className="text-[10px] text-gray-400">On-Chain Txs</div>
                </div>
                <div className="p-3 bg-[#0d1117] border border-green-500/30 rounded-xl text-center">
                  <div className="text-xl font-bold text-green-400">$0.01</div>
                  <div className="text-[10px] text-gray-400">Transfer</div>
                </div>
              </div>

              {/* Complete Verification Trail - Detailed like ACK */}
              <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                  <h3 className="text-sm font-semibold text-white">Complete Verification Trail</h3>
                </div>
                <div className="space-y-2">
                  {/* Step 1: Skyfire KYA */}
                  <div className="flex items-start gap-3 p-2 bg-orange-900/20 rounded-lg border border-orange-500/30">
                    <User className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <span className="text-orange-400 font-medium text-xs">1. Skyfire KYA: Agent identity verified</span>
                      <p className="text-gray-400 mt-0.5 text-[10px]">
                        Agent ID: <span className="text-orange-300 font-mono">{agent?.id || 'skyfire-demo'}</span>
                        {kyaToken && (
                          <span className="ml-2">• JWT signed by Skyfire</span>
                        )}
                      </p>
                    </div>
                  </div>

                  {/* Step 2: zkML Proof */}
                  <div className="flex items-start gap-3 p-2 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                    <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <span className="text-yellow-400 font-medium text-xs">2. zkML: Spending policy cryptographically verified</span>
                      <p className="text-gray-400 mt-0.5 text-[10px]">
                        Decision: <span className={spendingProof?.decision.shouldBuy ? 'text-green-400' : 'text-red-400'}>{spendingProof?.decision.shouldBuy ? 'APPROVE' : 'REJECT'}</span>
                        <span className="mx-1">•</span>
                        Confidence: <span className="text-yellow-300">{spendingProof ? (spendingProof.decision.confidence * 100).toFixed(0) : '95'}%</span>
                        <span className="mx-1">•</span>
                        Proof: <span className="text-yellow-300 font-mono">{spendingProof?.proofHash.slice(0, 10) || '0x...'}...</span>
                      </p>
                    </div>
                  </div>

                  {/* Step 3: Arc Attestation */}
                  <div className="flex items-start gap-3 p-2 bg-purple-900/20 rounded-lg border border-purple-500/30">
                    <Shield className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <span className="text-purple-400 font-medium text-xs">3. Arc: Verification result attested on-chain</span>
                      <p className="text-gray-400 mt-0.5 text-[10px]">
                        verificationHash bound to Skyfire agent ID
                        {attestationTxHash && (
                          <>
                            <span className="mx-1">•</span>
                            <a href={getExplorerTxUrl(attestationTxHash)} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                              View attestation tx ↗
                            </a>
                          </>
                        )}
                      </p>
                    </div>
                  </div>

                  {/* Step 4: Gated Transfer */}
                  <div className="flex items-start gap-3 p-2 bg-green-900/20 rounded-lg border border-green-500/30">
                    <CreditCard className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <span className="text-green-400 font-medium text-xs">4. Payment: Gated transfer executed via SpendingGateWallet</span>
                      <p className="text-gray-400 mt-0.5 text-[10px]">
                        Amount: <span className="text-green-300">$0.01 USDC</span>
                        <span className="mx-1">•</span>
                        Smart contract verified attestation before release
                        {txHash && (
                          <>
                            <span className="mx-1">•</span>
                            <a href={getExplorerTxUrl(txHash)} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                              View transfer tx ↗
                            </a>
                          </>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Summary */}
              {transferResult?.summary && (
                <div className="p-4 bg-orange-500/10 border border-orange-500/30 rounded-xl">
                  <h4 className="text-sm font-medium text-orange-400 mb-3">How zkML Extends Skyfire</h4>
                  <div className="space-y-2 text-xs text-gray-300">
                    <p><span className="text-orange-400">Skyfire provides:</span> Verified agent identity (KYA) and payment infrastructure</p>
                    <p><span className="text-yellow-400">zkML adds:</span> Cryptographic proof of spending policy compliance</p>
                    <p><span className="text-purple-400">Together:</span> Enterprise-grade agent commerce with auditable policy verification</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
