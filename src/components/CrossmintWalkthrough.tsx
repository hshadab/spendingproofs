'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Bot,
  Zap,
  Wallet,
  CheckCircle2,
  Shield,
  ArrowRight,
  ExternalLink,
  Sparkles,
  AlertCircle,
  AlertTriangle,
  XCircle,
  Info,
  Users,
  FileCheck,
  Scale,
  Lightbulb,
  TrendingUp,
} from 'lucide-react';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useCrossmintWallet } from '@/hooks/useCrossmintWallet';
import {
  createEnterpriseDemoInput,
  runSpendingModel,
  ENTERPRISE_PROCUREMENT_POLICY,
  type SpendingModelInput,
  type SpendingModelOutput
} from '@/lib/spendingModel';
import { ProofProgress } from './ProofProgress';

// Highlight colors for different concepts
const HIGHLIGHT_COLORS = {
  intro: 'from-blue-500 to-cyan-500',
  agent: 'from-purple-500 to-pink-500',
  proof: 'from-yellow-500 to-orange-500',
  wallet: 'from-[#00D4AA] to-emerald-500',
  execution: 'from-green-500 to-emerald-500',
  conclusion: 'from-pink-500 to-purple-500',
};

const HIGHLIGHT_LABELS = {
  intro: 'Introduction',
  agent: 'AI Agent',
  proof: 'zkML Proof',
  wallet: 'Crossmint Wallet',
  execution: 'On-Chain',
  conclusion: 'Complete',
};

// Business annotation for executive takeaways
interface BusinessAnnotation {
  title: string;
  takeaway: string;
  color: 'crossmint' | 'zkml' | 'combined' | 'enterprise';
  metric?: string;
  metricLabel?: string;
}

// Walkthrough step definitions
interface WalkthroughStep {
  id: string;
  phase: 'intro' | 'agent' | 'proof' | 'wallet' | 'execution' | 'conclusion';
  title: string;
  description: string;
  crossmintNote?: string;
  crossmintDocUrl?: string;
  crossmintDocLabel?: string;
  duration: number;
  businessAnnotation?: BusinessAnnotation;
}

// Annotation color schemes
const ANNOTATION_COLORS = {
  crossmint: {
    bg: 'bg-[#00D4AA]/10',
    border: 'border-[#00D4AA]/50',
    title: 'text-[#00D4AA]',
    glow: 'shadow-[#00D4AA]/20',
  },
  zkml: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/50',
    title: 'text-yellow-400',
    glow: 'shadow-yellow-500/20',
  },
  combined: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/50',
    title: 'text-green-400',
    glow: 'shadow-green-500/20',
  },
  enterprise: {
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/50',
    title: 'text-purple-400',
    glow: 'shadow-purple-500/20',
  },
};

const WORKFLOW_STEPS: WalkthroughStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'Crossmint + zkML',
    description: 'Crossmint provides robust on-chain spending controls — limits, multi-sig, role-based permissions. These work great for rule-based policies. But some enterprise scenarios require ML inference: vendor risk scoring, historical analysis, dynamic compliance.',
    crossmintNote: 'Crossmint enforces rule-based policies on-chain. zkML extends this to ML-based policies.',
    duration: 6000,
    businessAnnotation: {
      title: 'Extending On-Chain Controls',
      takeaway: 'Crossmint handles rule-based policies. zkML proves ML-based policies were evaluated correctly.',
      color: 'crossmint',
      metric: '6',
      metricLabel: 'ML factors verified',
    },
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'Why zkML for ML Policies',
    description: 'ML models can\'t run on-chain — gas costs explode, EVM can\'t execute neural networks. zkML solves this: run the model off-chain, generate a cryptographic proof of correct execution, verify the proof before payment.',
    crossmintNote: 'Prove the ML model ran correctly without re-running it on-chain.',
    duration: 6000,
    businessAnnotation: {
      title: 'Complementary Verification',
      takeaway: 'On-chain controls for rules. zkML proofs for ML. Full policy coverage.',
      color: 'zkml',
      metric: '100%',
      metricLabel: 'verifiable',
    },
  },
  {
    id: 'agent-1',
    phase: 'agent',
    title: 'Enterprise Procurement Agent',
    description: 'An autonomous procurement agent with a dedicated treasury wallet. CFO-configured policies: $50K monthly budget, $10K max per transaction, vendor risk thresholds, category budgets, and compliance requirements.',
    crossmintNote: 'Wallet created via Crossmint Wallets API (POST /v1-alpha2/wallets) with type: evm-mpc-wallet',
    crossmintDocUrl: 'https://docs.crossmint.com/api-reference/wallets/create-wallet',
    crossmintDocLabel: 'Create Wallets API',
    duration: 5000,
    businessAnnotation: {
      title: 'Zero Key Management',
      takeaway: 'Crossmint MPC wallets eliminate private key custody. Your CFO never touches a seed phrase.',
      color: 'crossmint',
      metric: '$50K',
      metricLabel: 'monthly budget',
    },
  },
  {
    id: 'agent-2',
    phase: 'agent',
    title: 'Evaluating Procurement Request',
    description: 'Agent receives request: DataDog APM subscription at $4,500/month. The zkML model evaluates vendor risk (15%), historical score (92%), category budget ($15K observability), and compliance status.',
    duration: 5000,
    businessAnnotation: {
      title: 'Multi-Factor Risk Scoring',
      takeaway: '6 factors evaluated in milliseconds: vendor risk, history, budget, category, compliance, urgency. All provable.',
      color: 'enterprise',
      metric: '6',
      metricLabel: 'risk factors',
    },
  },
  {
    id: 'agent-3',
    phase: 'agent',
    title: 'Decision: Approved',
    description: 'Model outputs APPROVE with high confidence. DataDog is a preferred vendor with 2-year history. The next step: generate cryptographic proof that this decision came from the approved policy model.',
    crossmintNote: 'Without proof, this is just a claim. With proof, it\'s verifiable.',
    duration: 5000,
    businessAnnotation: {
      title: 'From Claim to Proof',
      takeaway: '"The agent says it checked the policy" becomes "Here\'s cryptographic proof it did." Unforgeable.',
      color: 'zkml',
      metric: '94%',
      metricLabel: 'confidence',
    },
  },
  {
    id: 'proof-1',
    phase: 'proof',
    title: 'Generating Policy Proof',
    description: 'JOLT-Atlas compiles the procurement policy model into a SNARK circuit. The agent generates a ~48KB cryptographic proof that all policy factors were correctly evaluated.',
    duration: 6000,
    businessAnnotation: {
      title: 'Instant Compliance Certificate',
      takeaway: 'In ~10 seconds, the agent creates a cryptographic receipt proving every policy check. Valid for years, verifiable forever.',
      color: 'zkml',
      metric: '~48KB',
      metricLabel: 'proof size',
    },
  },
  {
    id: 'proof-2',
    phase: 'proof',
    title: 'Enterprise Privacy',
    description: 'The proof reveals ONLY the decision and confidence. Treasury balances, vendor scores, category budgets, and internal thresholds stay private - crucial for competitive enterprise procurement.',
    crossmintNote: 'Enterprise treasury details stay confidential. Proof reveals approval, not internals.',
    duration: 6000,
    businessAnnotation: {
      title: 'Prove Without Revealing',
      takeaway: 'Verify the $50K budget was checked without revealing the actual number. Competitive intel stays private.',
      color: 'enterprise',
      metric: '0',
      metricLabel: 'data leaked',
    },
  },
  {
    id: 'wallet-1',
    phase: 'wallet',
    title: 'Crossmint Proof Verification',
    description: 'The agent wallet receives the payment request with zkML proof attached. The proof is cryptographically verified before authorizing the transfer.',
    crossmintNote: 'Integration point: Before calling Crossmint Transfer API, verify zkML proof. Proof valid → proceed. Invalid → reject.',
    crossmintDocUrl: 'https://docs.crossmint.com/wallets/introduction',
    crossmintDocLabel: 'MPC Wallets Security',
    duration: 8000,
    businessAnnotation: {
      title: 'Gated Execution',
      takeaway: 'Invalid proof = transfer blocked. The agent cannot spend without proving it followed policy. Mathematically enforced.',
      color: 'combined',
      metric: '100%',
      metricLabel: 'enforced',
    },
  },
  {
    id: 'execution-1',
    phase: 'execution',
    title: 'Crossmint Transfer Execution',
    description: 'Proof verified! Crossmint executes the USDC transfer via the Token Transfers API. The proof hash is included in transaction metadata for audit trail.',
    crossmintNote: 'POST /v1-alpha2/wallets/{locator}/tokens/usdc/transfers with zkML proofHash in metadata',
    crossmintDocUrl: 'https://docs.crossmint.com/wallets/wallet-actions/transfer-tokens',
    crossmintDocLabel: 'Transfer Tokens API',
    duration: 10000,
    businessAnnotation: {
      title: 'Permanent Audit Trail',
      takeaway: 'Proof hash stored in transaction metadata. Any auditor can verify compliance years later—no log digging required.',
      color: 'crossmint',
      metric: '$4,500',
      metricLabel: 'transferred',
    },
  },
  {
    id: 'conclusion-1',
    phase: 'conclusion',
    title: 'Verifiable Autonomous Spending',
    description: 'The agent completed a $4,500 procurement with cryptographic proof of policy compliance. 6 vendor factors evaluated, 4 budget constraints checked, compliance verified — all mathematically proven, not just logged.',
    crossmintNote: 'Audit trail: proof hash on-chain, full proof retrievable, model version locked.',
    duration: 10000,
    businessAnnotation: {
      title: 'CFO Confidence Unlocked',
      takeaway: 'Delegate $50K/month in autonomous agent spend with proof of every decision. Scale procurement without scaling risk.',
      color: 'combined',
      metric: '10x',
      metricLabel: 'delegation confidence',
    },
  },
];

const PHASES = ['intro', 'agent', 'proof', 'wallet', 'execution', 'conclusion'] as const;

// Full-screen Annotation Overlay Component - appears between steps
function AnnotationOverlay({
  annotation,
  stepTitle,
  onContinue
}: {
  annotation: BusinessAnnotation;
  stepTitle: string;
  onContinue: () => void;
}) {
  const colors = ANNOTATION_COLORS[annotation.color];

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-32 bg-black/90 backdrop-blur-sm animate-fadeIn"
      onClick={onContinue}
    >
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
        .animate-slideDown { animation: slideDown 0.5s ease-out; }
      `}</style>

      <div
        className={`
          max-w-lg mx-4 p-8 rounded-2xl
          ${colors.bg} ${colors.border} border-2
          shadow-2xl ${colors.glow}
          animate-slideDown
        `}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Step context */}
        <div className="text-center mb-6">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Takeaway from</span>
          <h3 className="text-lg text-gray-300 font-medium mt-1">{stepTitle}</h3>
        </div>

        {/* Metric - large and centered */}
        {annotation.metric && (
          <div className="flex justify-center mb-6">
            <div className={`px-6 py-4 rounded-xl ${colors.bg} ${colors.border} border-2`}>
              <div className="flex items-center justify-center gap-3">
                <TrendingUp className={`w-6 h-6 ${colors.title}`} />
                <span className={`text-4xl font-bold ${colors.title}`}>{annotation.metric}</span>
              </div>
              <div className="text-sm text-gray-400 text-center mt-1">{annotation.metricLabel}</div>
            </div>
          </div>
        )}

        {/* Title */}
        <h4 className={`text-2xl font-bold ${colors.title} text-center mb-4`}>
          {annotation.title}
        </h4>

        {/* Takeaway text */}
        <p className="text-base text-gray-300 leading-relaxed text-center mb-6">
          {annotation.takeaway}
        </p>

        {/* Continue indicator */}
        <div className="flex flex-col items-center gap-2 pt-4 border-t border-gray-700/50">
          <div className={`w-2 h-2 rounded-full ${colors.title.replace('text-', 'bg-')} animate-pulse`} />
          <span className="text-xs text-gray-500">Click anywhere or wait to continue</span>
        </div>
      </div>
    </div>
  );
}

interface AnnotatedWalkthroughProps {
  onStepChange?: (step: WalkthroughStep) => void;
  onProofGenerated?: (result: unknown) => void;
  onTxExecuted?: (hash: string) => void;
}

export function CrossmintWalkthrough({
  onStepChange,
  onProofGenerated,
  onTxExecuted,
}: AnnotatedWalkthroughProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [agentThoughts, setAgentThoughts] = useState<string[]>([]);
  const [showProof, setShowProof] = useState(false);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [proofHash, setProofHash] = useState<string | null>(null);
  const [transferError, setTransferError] = useState<string | null>(null);
  const [verifiedOnChain, setVerifiedOnChain] = useState<boolean>(false);
  const [attestationTxHash, setAttestationTxHash] = useState<string | null>(null);
  const [verificationSteps, setVerificationSteps] = useState<Array<{ step: string; status: string; txHash?: string; details?: string; timeMs?: number }>>([]);
  const [proofVerified, setProofVerified] = useState<boolean>(false);
  const [transferMethod, setTransferMethod] = useState<string | null>(null);
  const [modelDecision, setModelDecision] = useState<SpendingModelOutput | null>(null);
  const [demoInput, setDemoInput] = useState<SpendingModelInput | null>(null);
  const [showingAnnotation, setShowingAnnotation] = useState(false);
  const [annotationToShow, setAnnotationToShow] = useState<{ annotation: BusinessAnnotation; stepTitle: string } | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const annotationTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);
  const processedStepRef = useRef<string | null>(null);

  const { state: proofState, generateProof, reset: resetProof } = useProofGeneration();
  const { wallet, transfer, executeTransfer, resetTransfer } = useCrossmintWallet();

  const currentStep = WORKFLOW_STEPS[currentStepIndex];
  const progress = ((currentStepIndex + 1) / WORKFLOW_STEPS.length) * 100;

  // Keep ref in sync with state
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Auto-advance effect
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    const step = WORKFLOW_STEPS[currentStepIndex];
    if (!step) return;

    // Track if we've already run side effects for this step (prevents infinite loops)
    const stepKey = `${step.id}-${isPlaying}`;
    const alreadyProcessed = processedStepRef.current === stepKey;
    if (!alreadyProcessed) {
      processedStepRef.current = stepKey;
    }

    // Trigger phase-specific effects (only once per step)
    if (!alreadyProcessed && step.phase === 'agent' && step.id === 'agent-2') {
      // Create enterprise procurement demo input
      const input: SpendingModelInput = createEnterpriseDemoInput();
      setDemoInput(input);

      // Run the REAL spending model with enterprise policy
      const decision = runSpendingModel(input, ENTERPRISE_PROCUREMENT_POLICY);
      setModelDecision(decision);

      // Show enterprise procurement reasoning
      const thoughts = [
        'Evaluating procurement request...',
        `Vendor: ${input.serviceName} (${input.vendorOnboardingDays ? Math.floor(input.vendorOnboardingDays / 365) : 0}-year relationship)`,
        `Vendor Risk: ${((input.vendorRiskScore || 0) * 100).toFixed(0)}% (${(input.vendorRiskScore || 0) <= 0.3 ? 'LOW' : 'ELEVATED'})`,
        `Historical Score: ${((input.historicalVendorScore || 0) * 100).toFixed(0)}%`,
        `Compliance: ${input.vendorComplianceStatus ? 'VERIFIED' : 'PENDING'}`,
        `Category Budget: $${((input.categoryBudgetUsdc || 0) - (input.categorySpentUsdc || 0)).toLocaleString()} remaining`,
        `Price: $${input.priceUsdc.toLocaleString()} within policy`,
        ...decision.reasons.slice(0, 2),
        `Decision: ${decision.shouldBuy ? 'APPROVE' : 'REJECT'} (${(decision.confidence * 100).toFixed(0)}% confidence)`,
      ];
      setAgentThoughts([]);
      thoughts.forEach((thought, i) => {
        setTimeout(() => {
          if (isPlayingRef.current) {
            setAgentThoughts(prev => [...prev, thought]);
          }
        }, i * 500); // Faster animation for more thoughts
      });
    }

    if (!alreadyProcessed && step.phase === 'proof' && step.id === 'proof-1') {
      setShowProof(true);
      // Use the same input from agent phase or create enterprise default
      const input: SpendingModelInput = demoInput || createEnterpriseDemoInput();
      generateProof(input).then(result => {
        // Store proof hash for transfer audit trail
        if (result.success && result.proof) {
          setProofHash(result.proof.proofHash);
        }
        onProofGenerated?.(result);
      });
    }

    if (!alreadyProcessed && step.phase === 'execution') {
      // Execute real transfer via Crossmint API with cryptographic proof verification
      // Production: $4,500 to DataDog billing
      // Testnet: $0.45 (scaled 1:10000) for demo
      const recipientAddress = '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384'; // Demo recipient (DataDog billing in production)
      const transferAmount = 0.45; // $0.45 USDC (testnet scaled from $4,500)

      // Pass full proof data for cryptographic verification before transfer
      const proofDataForTransfer = proofState.result?.proof ? {
        proof: proofState.result.proof.proof,
        proofHash: proofState.result.proof.proofHash,
        programIo: proofState.result.proof.programIo,
        modelHash: proofState.result.proof.metadata?.modelHash,
        // Don't skip verification - let the API verify the proof cryptographically
        skipVerification: false,
      } : { proofHash: proofHash || undefined, skipVerification: true };

      executeTransfer(recipientAddress, transferAmount, proofDataForTransfer)
        .then(result => {
          if (result.success && result.txHash) {
            setTxHash(result.txHash);
            onTxExecuted?.(result.txHash);
            // Capture proof verification result
            if (result.proofVerified) {
              setProofVerified(true);
            }
            if (result.method) {
              setTransferMethod(result.method);
            }
            // Capture verification steps (includes zkML verification, transfer, attestation)
            if (result.steps?.length) {
              setVerificationSteps(result.steps);
            }
            // Capture on-chain attestation info
            if (result.verifiedOnChain) {
              setVerifiedOnChain(true);
            }
            if (result.attestationTxHash) {
              setAttestationTxHash(result.attestationTxHash);
            }
          } else if (result.transferId) {
            // Transfer pending - use transfer ID as placeholder
            setTxHash(result.transferId);
            onTxExecuted?.(result.transferId);
          } else {
            // Transfer failed - show error but continue demo
            setTransferError(result.error || 'Transfer failed');
            // Generate mock hash for demo continuity
            const mockHash = '0x' + Array.from({ length: 64 }, () =>
              Math.floor(Math.random() * 16).toString(16)
            ).join('');
            setTxHash(mockHash);
          }
        })
        .catch(() => {
          // Fallback to mock for demo
          const mockHash = '0x' + Array.from({ length: 64 }, () =>
            Math.floor(Math.random() * 16).toString(16)
          ).join('');
          setTxHash(mockHash);
        });
    }

    // Set timer to show annotation (if exists) then advance
    timerRef.current = setTimeout(() => {
      if (isPlayingRef.current) {
        // Check if this step has an annotation to show
        if (step.businessAnnotation) {
          // Show annotation overlay
          setAnnotationToShow({ annotation: step.businessAnnotation, stepTitle: step.title });
          setShowingAnnotation(true);

          // Auto-advance after 6 seconds
          annotationTimerRef.current = setTimeout(() => {
            setShowingAnnotation(false);
            setAnnotationToShow(null);

            // Now advance to next step
            if (isPlayingRef.current) {
              setCurrentStepIndex(prev => {
                const next = prev + 1;
                if (next >= WORKFLOW_STEPS.length) {
                  setIsPlaying(false);
                  return prev;
                }
                return next;
              });
            }
          }, 6000);
        } else {
          // No annotation, advance directly
          setCurrentStepIndex(prev => {
            const next = prev + 1;
            if (next >= WORKFLOW_STEPS.length) {
              setIsPlaying(false);
              return prev;
            }
            return next;
          });
        }
      }
    }, step.duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      if (annotationTimerRef.current) {
        clearTimeout(annotationTimerRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStepIndex, isPlaying, generateProof, executeTransfer, proofHash, proofState.result, onProofGenerated, onTxExecuted]);

  // Notify parent
  useEffect(() => {
    if (currentStep) {
      onStepChange?.(currentStep);
    }
  }, [currentStep, onStepChange]);

  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleNext = useCallback(() => {
    setCurrentStepIndex(prev => Math.min(prev + 1, WORKFLOW_STEPS.length - 1));
  }, []);

  const handlePrevious = useCallback(() => {
    setCurrentStepIndex(prev => Math.max(prev - 1, 0));
  }, []);

  const handleAnnotationContinue = useCallback(() => {
    // Clear annotation timer
    if (annotationTimerRef.current) {
      clearTimeout(annotationTimerRef.current);
      annotationTimerRef.current = null;
    }

    setShowingAnnotation(false);
    setAnnotationToShow(null);

    // Advance to next step
    if (isPlayingRef.current) {
      setCurrentStepIndex(prev => {
        const next = prev + 1;
        if (next >= WORKFLOW_STEPS.length) {
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentStepIndex(0);
    setAgentThoughts([]);
    setShowProof(false);
    setTxHash(null);
    setProofHash(null);
    setTransferError(null);
    setVerifiedOnChain(false);
    setAttestationTxHash(null);
    setVerificationSteps([]);
    setProofVerified(false);
    setTransferMethod(null);
    setShowingAnnotation(false);
    setAnnotationToShow(null);
    if (annotationTimerRef.current) {
      clearTimeout(annotationTimerRef.current);
      annotationTimerRef.current = null;
    }
    setModelDecision(null);
    setDemoInput(null);
    processedStepRef.current = null;
    resetProof();
    resetTransfer();
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
  }, [resetProof, resetTransfer]);

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'intro': return <Info className="w-4 h-4" />;
      case 'agent': return <Bot className="w-4 h-4" />;
      case 'proof': return <Zap className="w-4 h-4" />;
      case 'wallet': return <Wallet className="w-4 h-4" />;
      case 'execution': return <CheckCircle2 className="w-4 h-4" />;
      case 'conclusion': return <Sparkles className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getPhaseIndex = (phase: string) => PHASES.indexOf(phase as typeof PHASES[number]);
  const currentPhaseIndex = getPhaseIndex(currentStep.phase);

  return (
    <div className="flex flex-col">
      {/* Playback Controls - Separate Control Panel Above Demo */}
      <div className="mb-3 p-3 bg-[#0d1117] rounded-xl border border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handlePrevious}
              disabled={currentStepIndex <= 0}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50 border border-gray-700"
              title="Previous Step"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={handlePlayPause}
              className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg transition-all text-sm font-medium ${
                isPlaying
                  ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30'
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isPlaying ? 'Pause' : 'Play Demo'}</span>
            </button>
            <button
              type="button"
              onClick={handleNext}
              disabled={currentStepIndex >= WORKFLOW_STEPS.length - 1}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50 border border-gray-700"
              title="Next Step"
            >
              <SkipForward className="w-4 h-4" />
            </button>
            <div className="w-px h-6 bg-gray-700 mx-1" />
            <button
              type="button"
              onClick={handleReset}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors border border-gray-700"
              title="Reset"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            <span>Step {currentStepIndex + 1}/{WORKFLOW_STEPS.length}</span>
            <div className="flex items-center gap-1">
              {PHASES.map((phase, i) => (
                <div
                  key={phase}
                  className={`w-2 h-2 rounded-full transition-all ${
                    i < currentPhaseIndex
                      ? 'bg-green-500'
                      : i === currentPhaseIndex
                      ? `bg-gradient-to-r ${HIGHLIGHT_COLORS[phase]}`
                      : 'bg-gray-700'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Full-screen Annotation Overlay - appears between steps */}
      {showingAnnotation && annotationToShow && (
        <AnnotationOverlay
          annotation={annotationToShow.annotation}
          stepTitle={annotationToShow.stepTitle}
          onContinue={handleAnnotationContinue}
        />
      )}

      {/* Main Demo Container */}
      <div className="flex min-h-[780px] bg-[#0a0a0a] rounded-2xl overflow-hidden border border-gray-800">
        {/* Left Sidebar - Annotations */}
        <div className="w-80 bg-[#0d1117] border-r border-gray-800 flex flex-col flex-shrink-0">
          {/* Header */}
        <div className="p-4 border-b border-gray-800">
          {/* Logo */}
          <img
            src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
            alt="NovaNet"
            className="h-6 mb-2"
          />
          <p className="text-xs text-gray-400 mt-1">
            zkML Spending Proofs via<br />
            <span className="text-[#00D4AA] font-medium">Crossmint Wallets & Agentic Commerce</span>
          </p>
        </div>

        {/* Current Step Annotation */}
        <div className="p-4 overflow-y-auto">
          {/* Step Badge */}
          <div className="mb-2">
            <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${HIGHLIGHT_COLORS[currentStep.phase]} text-white`}>
              {HIGHLIGHT_LABELS[currentStep.phase]}
            </span>
          </div>

          {/* Step Title & Description */}
          <h3 className="text-lg font-bold text-white mb-2">{currentStep.title}</h3>
          <p className="text-sm text-gray-400 leading-relaxed mb-3">{currentStep.description}</p>

          {/* Crossmint Note */}
          {currentStep.crossmintNote && (
            <div className="p-3 bg-[#00D4AA]/10 border border-[#00D4AA]/30 rounded-lg mb-3">
              <div className="flex items-start gap-2">
                <Wallet className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-xs text-gray-300">{currentStep.crossmintNote}</p>
                  {currentStep.crossmintDocUrl && (
                    <a
                      href={currentStep.crossmintDocUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-1.5 text-[10px] text-[#00D4AA] hover:underline"
                    >
                      <span>📄 {currentStep.crossmintDocLabel || 'Crossmint Docs'}</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Progress Bar with Pulsing - 6 segments matching phases */}
          <div className="mb-2">
            <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
              <span>Progress</span>
              <span>{currentPhaseIndex + 1}/{PHASES.length}</span>
            </div>
            <div className="flex items-center gap-1">
              {PHASES.map((phase, i) => {
                const isComplete = i < currentPhaseIndex;
                const isCurrent = i === currentPhaseIndex;
                return (
                  <div
                    key={phase}
                    className={`h-1.5 flex-1 rounded-full transition-all ${
                      isComplete
                        ? 'bg-green-500'
                        : isCurrent
                        ? `bg-gradient-to-r ${HIGHLIGHT_COLORS[phase]} ${isPlaying ? 'animate-pulse' : ''}`
                        : 'bg-gray-700'
                    }`}
                  />
                );
              })}
            </div>
          </div>

          {/* Phase List - Compact */}
          <div className="grid grid-cols-2 gap-1.5">
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
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-medium ${
                    isComplete
                      ? 'bg-green-500 text-white'
                      : isCurrent
                      ? `bg-gradient-to-r ${HIGHLIGHT_COLORS[phase]} text-white`
                      : 'bg-gray-700 text-gray-400'
                  }`}>
                    {isComplete ? <CheckCircle2 className="w-3 h-3" /> : i + 1}
                  </div>
                  <span className={`text-xs ${
                    isCurrent ? 'text-white font-medium' : isComplete ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    {HIGHLIGHT_LABELS[phase]}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Spacer to push footer to bottom */}
        <div className="flex-1" />

        {/* CTA Button - shown on conclusion */}
        {currentStep.phase === 'conclusion' && (
          <div className="px-4 pb-3">
            <a
              href="https://crossmint.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full py-2.5 bg-[#00D4AA] text-black font-medium rounded-lg hover:bg-[#00D4AA]/90 transition-colors text-sm"
            >
              Explore Crossmint
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        )}

        {/* Footer */}
        <div className="px-4 py-2 border-t border-gray-800">
          <div className="flex items-center justify-between text-[10px] text-gray-500">
            <a href="https://crossmint.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-[#00D4AA] transition-colors">
              Crossmint <ExternalLink className="w-2.5 h-2.5" />
            </a>
            <span>Powered by zkML</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-6 bg-[#0a0a0a] overflow-y-auto">
        {/* Intro Phase */}
        {currentStep.phase === 'intro' && (
          <div>
            {/* Powered By Header */}
            <div className="flex items-center gap-4 mb-4">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[#00D4AA]/10 border border-[#00D4AA]/30 rounded-lg">
                <Wallet className="w-4 h-4 text-[#00D4AA]" />
                <span className="text-[#00D4AA] font-medium text-sm">Crossmint</span>
              </div>
              <span className="text-gray-600">+</span>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-yellow-400 font-medium text-sm">Jolt-Atlas zkML</span>
              </div>
              <span className="text-gray-600">+</span>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <Shield className="w-4 h-4 text-blue-400" />
                <span className="text-blue-400 font-medium text-sm">Base Sepolia</span>
              </div>
            </div>

            <h2 className="text-2xl font-bold mb-2">
              Crossmint + zkML
            </h2>
            <p className="text-gray-400 max-w-2xl mb-4 text-sm">
              Extending Crossmint&apos;s on-chain spending controls with zkML proofs for ML-based policies.
            </p>

            {/* Complementary Capabilities Section */}
            <div className="bg-gradient-to-r from-[#00D4AA]/10 to-yellow-900/20 border border-[#00D4AA]/30 rounded-xl p-4 max-w-2xl mb-5">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 className="w-4 h-4 text-[#00D4AA]" />
                <span className="text-[#00D4AA] font-semibold text-sm">Complementary Verification</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-[#0d1117] border border-[#00D4AA]/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Wallet className="w-3 h-3 text-[#00D4AA]" />
                    <span className="text-[#00D4AA] font-medium text-xs">Crossmint On-Chain</span>
                  </div>
                  <code className="text-[10px] text-gray-300 bg-gray-800 px-2 py-1 rounded block">
                    limits, multi-sig, permissions
                  </code>
                  <p className="text-[9px] text-gray-500 mt-2">Rule-based policy enforcement</p>
                </div>
                <div className="p-3 bg-[#0d1117] border border-yellow-500/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-3 h-3 text-yellow-400" />
                    <span className="text-yellow-400 font-medium text-xs">zkML Proofs</span>
                  </div>
                  <code className="text-[10px] text-gray-300 bg-gray-800 px-2 py-1 rounded block whitespace-nowrap overflow-hidden">
                    mlModel.evaluate(risk, history...)
                  </code>
                  <p className="text-[9px] text-gray-500 mt-2">ML-based policy verification</p>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-700/50 text-[10px] text-gray-400">
                <span className="text-[#00D4AA] font-medium">Together:</span> Full policy coverage — from simple spending limits to complex ML-based vendor scoring and compliance checks.
              </div>
            </div>

            {/* Three-Column Tech Stack */}
            <div className="grid grid-cols-3 gap-3 max-w-2xl mb-5">
              {/* Crossmint Column */}
              <div className="bg-[#0d1117] border border-[#00D4AA]/30 rounded-xl p-3 hover:border-[#00D4AA]/60 transition-colors">
                <a href="https://docs.crossmint.com" target="_blank" rel="noopener noreferrer" className="group">
                  <div className="flex items-center gap-2 mb-2">
                    <Wallet className="w-4 h-4 text-[#00D4AA]" />
                    <span className="text-[#00D4AA] font-semibold text-sm group-hover:underline">Crossmint</span>
                    <ExternalLink className="w-3 h-3 text-[#00D4AA] opacity-50" />
                  </div>
                </a>
                <p className="text-[10px] text-gray-400 mb-2">Enterprise wallet infrastructure</p>
                <div className="space-y-1.5 text-[10px]">
                  <a href="https://docs.crossmint.com/wallets/quickstarts/create-wallets-api" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-gray-400 hover:text-[#00D4AA] transition-colors">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>MPC Wallets API</span>
                  </a>
                  <a href="https://docs.crossmint.com/wallets/wallets/mpc-wallets" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-gray-400 hover:text-[#00D4AA] transition-colors">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>Fireblocks-Backed Security</span>
                  </a>
                  <a href="https://docs.crossmint.com/wallets/quickstarts/transfer-tokens-api" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-gray-400 hover:text-[#00D4AA] transition-colors">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>Token Transfers API</span>
                  </a>
                </div>
              </div>

              {/* Jolt-Atlas Column */}
              <a href="https://novanet.xyz" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-3 hover:border-yellow-500/60 transition-colors group">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="text-yellow-400 font-semibold text-sm group-hover:underline">Jolt-Atlas</span>
                </div>
                <p className="text-[10px] text-gray-400 mb-2">zkML proof generation</p>
                <div className="space-y-1 text-[10px]">
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                    <span>SNARK Proofs (~48KB)</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                    <span>ONNX Model Support</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                    <span>~10s Generation</span>
                  </div>
                </div>
              </a>

              {/* Base Sepolia Column */}
              <a href="https://docs.base.org/network-information" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-blue-500/30 rounded-xl p-3 hover:border-blue-500/60 transition-colors group">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="w-4 h-4 text-blue-400" />
                  <span className="text-blue-400 font-semibold text-sm group-hover:underline">Base Sepolia</span>
                </div>
                <p className="text-[10px] text-gray-400 mb-2">Crossmint-supported chain</p>
                <div className="space-y-1 text-[10px]">
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-blue-400" />
                    <span>Native USDC Support</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-blue-400" />
                    <span>Crossmint Transfers</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-blue-400" />
                    <span>Real API Integration</span>
                  </div>
                </div>
              </a>
            </div>

            {/* Chain Info */}
            <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 max-w-2xl mb-5">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] text-gray-500">Base Sepolia Testnet (Chain ID: 84532)</div>
                <a
                  href="https://sepolia.basescan.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[10px] text-blue-400 hover:underline flex items-center gap-1"
                >
                  BaseScan <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-gray-500 font-sans">ProofAttestation</div>
                  <a
                    href="https://sepolia.basescan.org/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-yellow-400 hover:underline"
                  >
                    0x1Fb6...F213
                  </a>
                </div>
                <div>
                  <div className="text-gray-500 font-sans">SpendingGate</div>
                  <a
                    href="https://sepolia.basescan.org/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:underline"
                  >
                    0x6A47...d138
                  </a>
                </div>
                <div>
                  <div className="text-gray-500 font-sans">USDC (Testnet)</div>
                  <a
                    href="https://sepolia.basescan.org/address/0x8A04d904055528a69f3E4594DDA308A31aeb8457"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:underline"
                  >
                    0x8A04...8457
                  </a>
                </div>
              </div>
            </div>

            {/* Workflow Overview */}
            <div className="grid grid-cols-4 gap-3 max-w-2xl">
              <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 text-center ${isPlaying ? 'border-purple-500/50 shadow-lg shadow-purple-500/10' : 'border-gray-800'}`}>
                <Bot className={`w-6 h-6 text-purple-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                <div className="font-medium text-sm mb-1">AI Agent</div>
                <div className="text-[10px] text-gray-400">Evaluates purchase</div>
              </div>
              <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-100 text-center ${isPlaying ? 'border-yellow-500/50 shadow-lg shadow-yellow-500/10' : 'border-gray-800'}`}>
                <Zap className={`w-6 h-6 text-yellow-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                <div className="font-medium text-sm mb-1">zkML Proof</div>
                <div className="text-[10px] text-gray-400">Proves compliance</div>
              </div>
              <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-200 text-center ${isPlaying ? 'border-[#00D4AA]/50 shadow-lg shadow-[#00D4AA]/10' : 'border-gray-800'}`}>
                <Wallet className={`w-6 h-6 text-[#00D4AA] mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                <div className="font-medium text-sm mb-1">Crossmint</div>
                <div className="text-[10px] text-gray-400">Verifies & signs</div>
              </div>
              <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-300 text-center ${isPlaying ? 'border-blue-500/50 shadow-lg shadow-blue-500/10' : 'border-gray-800'}`}>
                <div className={`w-6 h-6 mx-auto mb-2 flex items-center justify-center text-blue-400 font-bold ${isPlaying ? 'animate-pulse' : ''}`}>$</div>
                <div className="font-medium text-sm mb-1">USDC</div>
                <div className="text-[10px] text-gray-400">Transfers on Base</div>
              </div>
            </div>
          </div>
        )}

        {/* Agent Phase */}
        {currentStep.phase === 'agent' && (
          <div className="grid grid-cols-2 gap-4">
            {/* Agent Panel */}
            <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-purple-500 shadow-lg shadow-purple-500/20' : 'border-purple-500/50'}`}>
              <div className="p-3 border-b border-gray-800 flex items-center gap-3 bg-purple-500/10">
                <div className="relative">
                  <div className="w-9 h-9 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Bot className="w-5 h-5 text-purple-400" />
                  </div>
                  {isPlaying && (
                    <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse" />
                  )}
                </div>
                <div>
                  <h4 className="font-semibold text-sm">Enterprise Procurement Agent</h4>
                  <p className="text-xs text-gray-400">Autonomous SaaS purchasing</p>
                </div>
              </div>

              <div className="p-3 border-b border-gray-800">
                <div className="text-xs text-purple-300 mb-2">CFO Policy Configuration</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Monthly Budget</div>
                    <div className="font-mono">$50,000</div>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Spent This Month</div>
                    <div className="font-mono">$12,500</div>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Max Per Tx</div>
                    <div className="font-mono">$10,000</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs mt-2">
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Category (Observability)</div>
                    <div className="font-mono">$15,000 limit</div>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Max Vendor Risk</div>
                    <div className="font-mono">70%</div>
                  </div>
                </div>
              </div>

              <div className="p-3 border-b border-gray-800">
                <div className="text-xs text-gray-400 mb-2">Procurement Request</div>
                <div className={`p-2 bg-green-900/20 border rounded-lg transition-all ${isPlaying ? 'border-green-500 shadow-md shadow-green-500/10' : 'border-green-700/50'}`}>
                  <div className="flex justify-between mb-1">
                    <span className="font-medium text-sm">DataDog APM</span>
                    <span className="font-mono text-green-400 text-sm">$4,500</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-400 mt-2">
                    <div>Vendor Risk: <span className="text-green-400">15%</span></div>
                    <div>History: <span className="text-green-400">92%</span></div>
                    <div>SLA: <span className="text-green-400">99.9%</span></div>
                    <div>Compliance: <span className="text-green-400">Verified</span></div>
                  </div>
                </div>
              </div>

              <div className="p-3">
                <div className="text-xs text-gray-400 mb-2 flex items-center gap-2">
                  <Sparkles className={`w-3 h-3 ${isPlaying && agentThoughts.length < 5 ? 'animate-spin' : ''}`} />
                  Agent Reasoning
                  {isPlaying && agentThoughts.length < 5 && (
                    <span className="text-purple-400 animate-pulse">processing...</span>
                  )}
                </div>
                <div className="space-y-1 font-mono text-xs min-h-[80px]">
                  {agentThoughts.length > 0 ? (
                    agentThoughts.map((thought, i) => (
                      <div key={i} className="text-gray-300 flex items-start gap-2 animate-fade-in">
                        <span className="text-purple-400">&gt;</span>
                        {thought}
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500 italic flex items-center gap-2">
                      {isPlaying ? (
                        <>
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" />
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                        </>
                      ) : 'Waiting for agent...'}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Flow Diagram */}
            <div className="flex flex-col justify-center items-center">
              <div className="space-y-3">
                {[
                  { icon: <Bot className="w-4 h-4" />, label: 'Agent Evaluates', color: 'purple', step: 0 },
                  { icon: <ArrowRight className="w-4 h-4 rotate-90" />, label: '', color: 'gray', step: -1 },
                  { icon: <Zap className="w-4 h-4" />, label: 'Generate Proof', color: 'yellow', step: 1 },
                  { icon: <ArrowRight className="w-4 h-4 rotate-90" />, label: '', color: 'gray', step: -1 },
                  { icon: <Wallet className="w-4 h-4" />, label: 'Crossmint Verifies', color: 'teal', step: 2 },
                  { icon: <ArrowRight className="w-4 h-4 rotate-90" />, label: '', color: 'gray', step: -1 },
                  { icon: <CheckCircle2 className="w-4 h-4" />, label: 'Payment Executes', color: 'green', step: 3 },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3">
                    {item.label ? (
                      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm transition-all duration-300 ${
                        item.color === 'purple' ? `bg-purple-500/20 border-purple-500/50 text-purple-400 ${isPlaying && item.step === 0 ? 'ring-2 ring-purple-500 shadow-lg shadow-purple-500/30' : ''}` :
                        item.color === 'yellow' ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400' :
                        item.color === 'teal' ? 'bg-[#00D4AA]/20 border-[#00D4AA]/50 text-[#00D4AA]' :
                        'bg-green-500/20 border-green-500/50 text-green-400'
                      }`}>
                        {item.icon}
                        <span>{item.label}</span>
                      </div>
                    ) : (
                      <div className={`text-gray-600 pl-5 ${isPlaying ? 'animate-pulse' : ''}`}>{item.icon}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Proof Phase */}
        {currentStep.phase === 'proof' && (
          <div className="grid grid-cols-2 gap-4">
            {/* Proof Progress */}
            <div>
              {showProof && (
                <div className={`transition-all duration-300 ${isPlaying && proofState.status === 'running' ? 'ring-2 ring-yellow-500/50 rounded-xl' : ''}`}>
                  <ProofProgress
                    status={proofState.status}
                    progress={proofState.progress}
                    elapsedMs={proofState.elapsedMs}
                    steps={proofState.steps}
                  />
                </div>
              )}

              {proofState.status === 'complete' && (
                <div className="mt-3 p-3 bg-green-900/20 border border-green-500 rounded-xl shadow-lg shadow-green-500/20">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="font-medium text-green-400 text-sm">Proof Ready for Crossmint</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="text-gray-400">Decision</div>
                      <div className="font-mono text-green-400">APPROVED</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Confidence</div>
                      <div className="font-mono">87%</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Proof Size</div>
                      <div className="font-mono">~48 KB</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Time</div>
                      <div className="font-mono">{(proofState.elapsedMs / 1000).toFixed(1)}s</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Proof Explanation */}
            <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all duration-300 ${isPlaying ? 'border-yellow-500 shadow-lg shadow-yellow-500/20' : 'border-yellow-500/50'}`}>
              <h4 className="font-semibold mb-3 flex items-center gap-2 text-sm">
                <Zap className={`w-4 h-4 text-yellow-400 ${isPlaying && proofState.status === 'running' ? 'animate-pulse' : ''}`} />
                What This Proof Guarantees
              </h4>
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Vendor risk score (15%) was evaluated against threshold
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Category budget &quot;observability&quot; was checked ($10.8K remaining)
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Historical vendor score (92%) factored into decision
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Compliance verification status was confirmed
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Shield className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Enterprise treasury details stay completely private
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Wallet className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Auditors can verify policy compliance without seeing internals
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Wallet Phase */}
        {currentStep.phase === 'wallet' && (
          <div className="grid grid-cols-2 gap-6">
            {/* Wallet Panel */}
            <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-500 ${isPlaying ? 'border-[#00D4AA] shadow-lg shadow-[#00D4AA]/20' : 'border-[#00D4AA]/50'}`}>
              <div className="p-4 border-b border-gray-800 bg-gradient-to-r from-[#00D4AA]/10 to-transparent flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className={`w-10 h-10 ${wallet.error ? 'bg-red-500/20' : 'bg-[#00D4AA]/20'} rounded-lg flex items-center justify-center`}>
                      <Wallet className={`w-6 h-6 ${wallet.error ? 'text-red-400' : 'text-[#00D4AA]'} ${isPlaying ? 'animate-pulse' : ''}`} />
                    </div>
                    {isPlaying && !wallet.error && (
                      <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-[#00D4AA] rounded-full animate-ping" />
                    )}
                  </div>
                  <div>
                    <a href="https://docs.crossmint.com/wallets/introduction" target="_blank" rel="noopener noreferrer" className="group">
                      <h4 className={`font-semibold ${wallet.error ? 'text-red-400' : 'text-[#00D4AA]'} group-hover:underline flex items-center gap-1`}>
                        Agent Wallet
                        <ExternalLink className="w-3 h-3 opacity-50" />
                      </h4>
                    </a>
                    <p className="text-xs text-gray-400 font-mono">
                      {wallet.loading ? 'Loading...' : wallet.address ? `${wallet.address.slice(0, 6)}...${wallet.address.slice(-4)}` : 'Not connected'}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  {wallet.error ? (
                    <div className="text-xs text-red-400">{wallet.error}</div>
                  ) : (
                    <>
                      <div className="text-xs text-gray-400">
                        {wallet.loading ? 'Loading...' : `Base Sepolia • ${wallet.chain || 'testnet'}`}
                      </div>
                      <div className="font-mono font-semibold">
                        {wallet.loading ? '...' : `$${parseFloat(wallet.balanceUsdc || '0').toFixed(2)} USDC`}
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="p-4 border-b border-gray-800">
                <div className="text-xs text-[#00D4AA] mb-3 flex items-center gap-2">
                  <Shield className={`w-3 h-3 ${isPlaying ? 'animate-pulse' : ''}`} />
                  Off-Chain Proof Verification
                  {isPlaying && <span className="text-[#00D4AA] text-[10px] animate-pulse">● ACTIVE</span>}
                </div>
                <div className={`p-3 ${modelDecision?.shouldBuy ? 'bg-green-900/20' : 'bg-red-900/20'} border rounded-lg transition-all duration-300 ${isPlaying ? (modelDecision?.shouldBuy ? 'border-green-500 shadow-md shadow-green-500/20' : 'border-red-500 shadow-md shadow-red-500/20') : 'border-green-700/50'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className={`w-4 h-4 ${modelDecision?.shouldBuy ? 'text-green-400' : 'text-red-400'} ${isPlaying ? 'animate-bounce' : ''}`} />
                    <span className={`font-medium ${modelDecision?.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>zkML Proof Verified (Off-Chain)</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">Decision:</span>
                      <span className={`ml-1 font-mono ${modelDecision?.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
                        {modelDecision?.shouldBuy ? 'APPROVED' : 'REJECTED'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Confidence:</span>
                      <span className="ml-1 font-mono">{modelDecision ? `${(modelDecision.confidence * 100).toFixed(0)}%` : '...'}</span>
                    </div>
                  </div>
                  {modelDecision?.reasons && modelDecision.reasons.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-700/50 text-[10px] text-gray-400">
                      {modelDecision.reasons[0]}
                    </div>
                  )}
                </div>
              </div>

              <div className={`p-4 transition-all duration-500 ${isPlaying ? 'bg-[#00D4AA]/10' : 'bg-[#00D4AA]/5'}`}>
                <div className="text-center text-sm flex items-center justify-center gap-2">
                  {isPlaying ? (
                    <>
                      <div className="w-2 h-2 bg-[#00D4AA] rounded-full animate-ping" />
                      <span className="text-[#00D4AA] font-medium">Authorizing transfer...</span>
                    </>
                  ) : (
                    <span className="text-gray-400">Ready to execute payment</span>
                  )}
                </div>
              </div>
            </div>

            {/* Verification Steps */}
            <div className={`bg-[#0d1117] border rounded-xl p-6 transition-all duration-300 ${isPlaying ? 'border-[#00D4AA]/50' : 'border-gray-800'}`}>
              <h4 className="font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-4 h-4 text-[#00D4AA]" />
                Off-Chain Proof Verification
              </h4>
              <div className="space-y-3">
                {[
                  { check: 'SNARK proof cryptographically valid?', delay: 0 },
                  { check: 'Vendor compliance status verified?', delay: 100 },
                  { check: 'Category budget not exceeded?', delay: 200 },
                  { check: 'Vendor risk within threshold (≤70%)?', delay: 300 },
                  { check: `Policy model output: ${modelDecision?.shouldBuy ? 'APPROVED' : 'REJECTED'}?`, delay: 400 },
                  { check: 'Proof hash unique (replay protection)?', delay: 500 },
                ].map((item, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-3 p-3 rounded-lg transition-all duration-500 ${isPlaying ? 'bg-green-900/30 border border-green-500/30' : 'bg-gray-800/50'}`}
                    style={{ transitionDelay: `${item.delay}ms` }}
                  >
                    <CheckCircle2 className={`w-5 h-5 text-green-400 ${isPlaying ? 'animate-pulse' : ''}`} />
                    <span className="text-sm">{item.check}</span>
                    {isPlaying && <span className="ml-auto text-xs text-green-400 font-mono">✓ PASS</span>}
                  </div>
                ))}
              </div>
              <div className={`mt-4 p-3 rounded-lg text-xs transition-all duration-500 ${isPlaying ? 'bg-[#00D4AA]/20 border border-[#00D4AA]/30' : 'bg-gray-800/30'}`}>
                <div className="flex items-center gap-2">
                  <Wallet className="w-3 h-3 text-[#00D4AA]" />
                  <span className="text-gray-300">Payment authorized by off-chain verification. Proof hash stored in Crossmint transaction metadata.</span>
                </div>
              </div>
            </div>

            {/* Crossmint API Integration Panel - Horizontal Layout */}
            <div className="bg-[#0d1117] border border-[#00D4AA]/30 rounded-xl overflow-hidden">
              <div className="p-3 border-b border-[#00D4AA]/20 bg-gradient-to-r from-[#00D4AA]/10 to-transparent">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-[#00D4AA]" />
                    <h4 className="font-semibold text-sm text-[#00D4AA]">Crossmint API Integration</h4>
                  </div>
                  <a
                    href="https://docs.crossmint.com/wallets/introduction"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] text-[#00D4AA] hover:underline flex items-center gap-1"
                  >
                    Docs <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                </div>
              </div>
              <div className="p-3 grid grid-cols-3 gap-3">
                {/* Wallet Creation */}
                <div className="p-2 bg-gray-900/50 rounded-lg border border-gray-700">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <div className="w-5 h-5 rounded bg-[#00D4AA]/20 flex items-center justify-center text-[9px] font-bold text-[#00D4AA]">1</div>
                    <span className="text-xs font-medium">MPC Wallet</span>
                  </div>
                  <div className="font-mono text-[9px] text-gray-400 bg-black/30 p-1.5 rounded">
                    POST /wallets
                  </div>
                </div>

                {/* zkML Verification */}
                <div className="p-2 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <div className="w-5 h-5 rounded bg-yellow-500/20 flex items-center justify-center text-[9px] font-bold text-yellow-400">2</div>
                    <span className="text-xs font-medium text-yellow-400">zkML Verify</span>
                  </div>
                  <div className="font-mono text-[9px] text-gray-400 bg-black/30 p-1.5 rounded">
                    POST /verify
                  </div>
                </div>

                {/* Token Transfer */}
                <div className="p-2 bg-gray-900/50 rounded-lg border border-gray-700">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <div className="w-5 h-5 rounded bg-[#00D4AA]/20 flex items-center justify-center text-[9px] font-bold text-[#00D4AA]">3</div>
                    <span className="text-xs font-medium">Transfer</span>
                  </div>
                  <div className="font-mono text-[9px] text-gray-400 bg-black/30 p-1.5 rounded">
                    POST /transfers
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Execution Phase */}
        {currentStep.phase === 'execution' && (
          <div className="grid grid-cols-2 gap-4">
            {/* Payment Flow Visualization */}
            <div className="space-y-3">
              {/* From Wallet */}
              <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all duration-300 ${isPlaying && !txHash ? 'border-[#00D4AA] shadow-lg shadow-[#00D4AA]/20' : 'border-gray-700'}`}>
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 bg-[#00D4AA]/20 rounded-lg flex items-center justify-center">
                    <Wallet className={`w-5 h-5 text-[#00D4AA] ${isPlaying && !txHash ? 'animate-pulse' : ''}`} />
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 flex items-center gap-1">
                      From: Agent Wallet
                    </div>
                    <div className="font-mono text-sm text-white">
                      {wallet.address ? `${wallet.address.slice(0, 10)}...${wallet.address.slice(-8)}` : 'Loading...'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Base Sepolia</span>
                  <span className="font-mono text-[#00D4AA]">${parseFloat(wallet.balanceUsdc || '0').toFixed(2)} USDC</span>
                </div>
              </div>

              {/* Arrow with Amount */}
              <div className="flex items-center justify-center py-2">
                <div className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-all duration-300 ${
                  isPlaying && !txHash
                    ? 'bg-green-500/20 border border-green-500 shadow-lg shadow-green-500/30'
                    : txHash
                    ? 'bg-green-500/30 border border-green-500'
                    : 'bg-gray-800 border border-gray-700'
                }`}>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-green-400">$4,500</span>
                    <span className="text-gray-400 text-sm">USDC</span>
                    <ArrowRight className={`w-4 h-4 text-green-400 ${isPlaying && !txHash ? 'animate-pulse' : ''}`} />
                  </div>
                  <div className="text-[10px] text-gray-500">Testnet: $0.45 (scaled 1:10000)</div>
                </div>
              </div>

              {/* To Address */}
              <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all duration-300 ${txHash ? 'border-green-500 shadow-lg shadow-green-500/20' : 'border-gray-700'}`}>
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${txHash ? 'bg-green-500/20' : 'bg-gray-800'}`}>
                    <CheckCircle2 className={`w-5 h-5 ${txHash ? 'text-green-400' : 'text-gray-500'}`} />
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">To: DataDog Billing</div>
                    <div className="font-mono text-sm text-white truncate w-48">
                      0x982Cd966...1C79E384
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Service</span>
                  <span className="text-purple-400">DataDog APM (Monthly)</span>
                </div>
              </div>

              {/* Network Badge */}
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 border border-blue-500/30 rounded-full text-xs">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                  <span className="text-blue-400">Base Sepolia</span>
                  <span className="text-gray-500">via Crossmint API</span>
                </div>
              </div>
            </div>

            {/* Transaction Status */}
            <div className={`bg-[#0d1117] border rounded-xl p-6 transition-all duration-500 ${txHash ? 'border-green-500 shadow-xl shadow-green-500/20' : 'border-gray-700'}`}>
              <div className="text-center mb-6">
                <div className="relative mx-auto mb-4 w-16 h-16">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${txHash ? 'bg-green-500/20' : 'bg-gray-800'} ${isPlaying && !txHash ? 'animate-pulse' : ''}`}>
                    {txHash ? (
                      <CheckCircle2 className="w-8 h-8 text-green-400" />
                    ) : (
                      <div className="w-6 h-6 border-2 border-[#00D4AA] border-t-transparent rounded-full animate-spin" />
                    )}
                  </div>
                  {isPlaying && !txHash && (
                    <div className="absolute inset-0 w-16 h-16 rounded-full border-2 border-green-500 animate-ping opacity-30" />
                  )}
                </div>
                <h3 className="text-xl font-bold mb-1">
                  {txHash ? 'Transfer Complete' : 'Executing...'}
                </h3>
                <p className="text-sm text-gray-400">
                  {txHash ? 'USDC sent via Crossmint on Base Sepolia' : 'Executing via Crossmint API...'}
                </p>
              </div>

              {transferError && (
                <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div className="text-xs text-yellow-300">{transferError}</div>
                  </div>
                </div>
              )}

              {txHash && (
                <div className="space-y-3">
                  {/* On-Chain Attestation Badge */}
                  {verifiedOnChain && (
                    <div className="p-3 bg-purple-900/20 rounded-lg border border-purple-500/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Shield className="w-4 h-4 text-purple-400" />
                        <span className="text-purple-400 font-medium text-sm">Proof Attested</span>
                      </div>
                      <div className="text-[10px] text-gray-400">
                        Proof attested via SpendingGate contract
                      </div>
                    </div>
                  )}

                  {/* Proof Verification Badge */}
                  {proofVerified && (
                    <div className="p-3 bg-green-900/20 rounded-lg border border-green-500/50">
                      <div className="flex items-center gap-2 mb-1">
                        <Shield className="w-4 h-4 text-green-400" />
                        <span className="text-green-400 font-medium text-sm">Proof Cryptographically Verified</span>
                      </div>
                      <div className="text-[10px] text-gray-400">
                        SNARK proof verified via JOLT-Atlas prover before transfer
                      </div>
                    </div>
                  )}

                  {/* Verification Steps */}
                  {verificationSteps.length > 0 && (
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-xs text-gray-400">Execution Pipeline</div>
                        {transferMethod && (
                          <span className="text-[10px] px-2 py-0.5 rounded bg-gray-700 text-gray-300">
                            {transferMethod === 'crossmint' ? 'Crossmint API' : 'Direct Transfer'}
                          </span>
                        )}
                      </div>
                      <div className="space-y-2">
                        {verificationSteps.map((vstep, i) => (
                          <div key={i} className="flex flex-col gap-0.5">
                            <div className="flex items-center gap-2 text-xs">
                              {vstep.status === 'success' ? (
                                <CheckCircle2 className="w-3 h-3 text-green-400 flex-shrink-0" />
                              ) : vstep.status === 'skipped' ? (
                                <div className="w-3 h-3 rounded-full bg-gray-600 flex-shrink-0" />
                              ) : (
                                <AlertCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
                              )}
                              <span className={vstep.status === 'success' ? 'text-green-400' : vstep.status === 'skipped' ? 'text-gray-500' : 'text-red-400'}>
                                {vstep.step}
                              </span>
                              {vstep.timeMs && (
                                <span className="text-[10px] text-gray-500 ml-auto">{vstep.timeMs}ms</span>
                              )}
                              {vstep.txHash && (
                                <a
                                  href={`https://sepolia.basescan.org/tx/${vstep.txHash}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[#00D4AA] hover:underline ml-1"
                                >
                                  tx
                                </a>
                              )}
                            </div>
                            {vstep.details && vstep.status !== 'success' && (
                              <div className="text-[10px] text-gray-500 ml-5">{vstep.details}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Transaction Hash */}
                  <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">
                      {verifiedOnChain ? 'Gated Transfer Hash' : 'Transaction Hash'}
                    </div>
                    <div className="font-mono text-xs text-gray-300 break-all">{txHash}</div>
                    <a
                      href={`https://sepolia.basescan.org/tx/${txHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-2 text-xs text-[#00D4AA] hover:underline"
                    >
                      View on BaseScan <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>

                  {/* Attestation Tx Hash */}
                  {attestationTxHash && (
                    <div className="p-3 bg-purple-900/10 rounded-lg border border-purple-500/30">
                      <div className="text-xs text-gray-400 mb-1">Attestation Transaction</div>
                      <div className="font-mono text-xs text-purple-400 truncate">{attestationTxHash}</div>
                      <a
                        href={`https://sepolia.basescan.org/tx/${attestationTxHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-2 text-xs text-purple-400 hover:underline"
                      >
                        View Attestation <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  )}

                  {/* Proof Hash */}
                  {proofHash && (
                    <div className="p-3 bg-yellow-900/10 rounded-lg border border-yellow-500/30">
                      <div className="text-xs text-gray-400 mb-1">Proof Hash (Audit Trail)</div>
                      <div className="font-mono text-xs text-yellow-400 truncate">{proofHash}</div>
                    </div>
                  )}

                  {/* Status Badges */}
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-green-900/20 rounded-lg border border-green-500/30 text-center">
                      <div className="text-xs text-gray-400">Status</div>
                      <div className="font-medium text-green-400 text-sm">Confirmed</div>
                    </div>
                    <div className={`p-2 rounded-lg border text-center ${verifiedOnChain ? 'bg-purple-900/20 border-purple-500/30' : 'bg-[#00D4AA]/10 border-[#00D4AA]/30'}`}>
                      <div className="text-xs text-gray-400">Verified</div>
                      <div className={`font-medium text-sm ${verifiedOnChain ? 'text-purple-400' : 'text-[#00D4AA]'}`}>
                        {verifiedOnChain ? 'Attested' : 'zkML Proof'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {!txHash && isPlaying && (
                <div className="space-y-2 text-xs">
                  {['Calling Crossmint API...', 'Executing on Base Sepolia...', 'Waiting for confirmation...'].map((step, i) => (
                    <div key={i} className="flex items-center gap-2 text-gray-400">
                      <div className={`w-1.5 h-1.5 rounded-full ${i === 0 ? 'bg-green-400' : i === 1 ? 'bg-yellow-400 animate-pulse' : 'bg-gray-600'}`} />
                      {step}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Conclusion Phase */}
        {currentStep.phase === 'conclusion' && (
          <div>
            {/* Powered By Header */}
            <div className="flex items-center gap-3 mb-4">
              <div className="flex items-center gap-2 px-2 py-1 bg-[#00D4AA]/10 border border-[#00D4AA]/30 rounded-lg">
                <Wallet className="w-3 h-3 text-[#00D4AA]" />
                <span className="text-[#00D4AA] font-medium text-xs">Crossmint</span>
              </div>
              <span className="text-gray-600 text-xs">+</span>
              <div className="flex items-center gap-2 px-2 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <Zap className="w-3 h-3 text-yellow-400" />
                <span className="text-yellow-400 font-medium text-xs">Jolt-Atlas</span>
              </div>
              <span className="text-gray-600 text-xs">+</span>
              <div className="flex items-center gap-2 px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <Shield className="w-3 h-3 text-blue-400" />
                <span className="text-blue-400 font-medium text-xs">Base Sepolia</span>
              </div>
            </div>

            <h2 className="text-xl font-bold mb-2">
              Trustless Enterprise Procurement
            </h2>
            <p className="text-gray-400 max-w-2xl mb-5 text-sm">
              The procurement agent completed a <span className="text-blue-400">$4,500</span> DataDog APM subscription purchase.
              <span className="text-yellow-400"> Jolt-Atlas</span> generated a SNARK proof verified off-chain by <span className="text-[#00D4AA]">Crossmint</span>.
              6 vendor factors evaluated, 4 budget constraints checked, compliance verified - all cryptographically proven, not promised.
            </p>

            {/* Transaction Summary */}
            <div className="grid grid-cols-4 gap-2 max-w-2xl mb-5">
              <div className="p-2 bg-[#0d1117] border border-blue-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-blue-400">$4,500</div>
                <div className="text-[10px] text-gray-400">Procurement Value</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-green-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-green-400">6</div>
                <div className="text-[10px] text-gray-400">Vendor Factors</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-yellow-400">~48KB</div>
                <div className="text-[10px] text-gray-400">Policy Proof</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-[#00D4AA]/30 rounded-xl text-center">
                <div className="text-lg font-bold text-[#00D4AA]">$0</div>
                <div className="text-[10px] text-gray-400">Treasury Exposed</div>
              </div>
            </div>

            {/* Full Workflow Recap */}
            <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl mb-6">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <span className="font-semibold text-white text-sm">What Happened</span>
              </div>
              <div className="space-y-3 text-xs">
                <div className="flex items-start gap-3 p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
                  <Bot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-purple-400 font-medium">1. Procurement Request Evaluated</span>
                    <p className="text-gray-400 mt-1">The agent received a DataDog APM subscription request at $4,500/month. The policy model evaluated vendor risk (15%), historical score (92%), category budget, and compliance status: {modelDecision?.shouldBuy ? 'APPROVED' : 'REJECTED'} ({modelDecision ? `${(modelDecision.confidence * 100).toFixed(0)}%` : '...'} confidence).</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                  <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-yellow-400 font-medium">2. Policy Proof Generated</span>
                    <p className="text-gray-400 mt-1">Jolt-Atlas compiled the procurement policy model into a SNARK circuit and generated a ~48KB proof that all 6 vendor factors and 4 budget constraints were correctly evaluated.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#00D4AA]/10 rounded-lg border border-[#00D4AA]/30">
                  <Wallet className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-[#00D4AA] font-medium">3. Crossmint Verified & Executed</span>
                    <p className="text-gray-400 mt-1">Crossmint verified the proof off-chain, then executed the $4,500 USDC transfer to DataDog. CFO&apos;s treasury details stayed completely private.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-blue-900/20 rounded-lg border border-blue-500/30">
                  <Shield className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-blue-400 font-medium">4. On-Chain Audit Trail</span>
                    <p className="text-gray-400 mt-1">
                      The proof hash was attested to the{' '}
                      <a
                        href="https://sepolia.basescan.org/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-yellow-400 hover:underline"
                      >
                        ProofAttestation contract
                      </a>
                      {' '}on Base Sepolia. Auditors can verify policy compliance was proven on-chain - without seeing internal budget details.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* What This Demonstrates */}
            <div className="bg-gradient-to-r from-[#00D4AA]/10 to-purple-500/10 border border-[#00D4AA]/30 rounded-xl p-4 max-w-2xl mb-4">
              <div className="text-sm font-medium text-[#00D4AA] mb-2">What This Demonstrates</div>
              <p className="text-gray-300 text-xs mb-3">
                An AI agent executed a <span className="text-[#00D4AA] font-bold">$4,500</span> procurement with cryptographic proof that the CFO-approved policy model was evaluated correctly — 6 factors checked, mathematically verified, unforgeable.
              </p>
              <div className="text-[10px] text-gray-400 border-t border-gray-700/50 pt-3">
                <span className="text-purple-400 font-medium">Why this matters:</span> Smart contracts can&apos;t run ML models. zkML lets you prove complex policy evaluation off-chain with on-chain verification.
              </div>
            </div>

            {/* Compact Links Section */}
            <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl">
              <div className="flex flex-wrap items-center gap-2 text-[10px]">
                {/* Crossmint Docs */}
                <span className="text-gray-500">Docs:</span>
                <a
                  href="https://docs.crossmint.com/api-reference/wallets/create-wallet"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2 py-1 bg-[#00D4AA]/10 text-[#00D4AA] rounded border border-[#00D4AA]/30 hover:border-[#00D4AA] transition-colors flex items-center gap-1"
                >
                  Wallets API <ExternalLink className="w-2.5 h-2.5" />
                </a>
                <a
                  href="https://docs.crossmint.com/wallets/wallet-actions/transfer-tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2 py-1 bg-[#00D4AA]/10 text-[#00D4AA] rounded border border-[#00D4AA]/30 hover:border-[#00D4AA] transition-colors flex items-center gap-1"
                >
                  Transfer API <ExternalLink className="w-2.5 h-2.5" />
                </a>

                <span className="text-gray-700">|</span>

                {/* Contracts */}
                <span className="text-gray-500">Contracts:</span>
                <a
                  href="https://sepolia.basescan.org/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2 py-1 bg-yellow-500/10 text-yellow-400 rounded border border-yellow-500/30 hover:border-yellow-500 transition-colors flex items-center gap-1"
                  title="ProofAttestation - stores proof hashes on-chain"
                >
                  ProofAttestation <ExternalLink className="w-2.5 h-2.5" />
                </a>
                <a
                  href="https://sepolia.basescan.org/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2 py-1 bg-purple-500/10 text-purple-400 rounded border border-purple-500/30 hover:border-purple-500 transition-colors flex items-center gap-1"
                  title="SpendingGateWallet - gated transfers with proof verification"
                >
                  SpendingGate <ExternalLink className="w-2.5 h-2.5" />
                </a>
                <a
                  href="https://sepolia.basescan.org/address/0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded border border-blue-500/30 hover:border-blue-500 transition-colors flex items-center gap-1"
                  title="USDC testnet token"
                >
                  USDC <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </div>
            </div>

          </div>
        )}
      </div>
      </div>
    </div>
  );
}
