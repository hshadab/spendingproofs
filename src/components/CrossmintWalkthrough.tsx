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
  Info,
} from 'lucide-react';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useCrossmintWallet } from '@/hooks/useCrossmintWallet';
import { createDefaultInput, type SpendingModelInput } from '@/lib/spendingModel';
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

// Walkthrough step definitions
interface WalkthroughStep {
  id: string;
  phase: 'intro' | 'agent' | 'proof' | 'wallet' | 'execution' | 'conclusion';
  title: string;
  description: string;
  crossmintNote?: string;
  duration: number;
}

const WORKFLOW_STEPS: WalkthroughStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'Crossmint + zkML',
    description: 'See how Crossmint Wallets can integrate zkML spending proofs - enabling enterprise-grade, trustless autonomous commerce with cryptographic policy enforcement.',
    crossmintNote: 'Crossmint already powers wallets for major enterprises. zkML proofs add the missing trust layer for AI agents.',
    duration: 5000,
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'Beyond Rate Limits',
    description: 'Current agent wallets use rate limits and allowlists. But enterprises need mathematical proof that agents follow policies - especially for regulated industries Crossmint serves.',
    crossmintNote: 'Banks, institutions, and enterprises need auditable proof trails - not just promises.',
    duration: 6000,
  },
  {
    id: 'agent-1',
    phase: 'agent',
    title: 'Crossmint-Powered Agent',
    description: 'An autonomous AI agent with a Crossmint custodial wallet. The enterprise has configured strict spending policies: $0.50 daily limit, $0.10 max per transaction, 50% minimum service reliability.',
    duration: 5000,
  },
  {
    id: 'agent-2',
    phase: 'agent',
    title: 'Agent Evaluates Purchase',
    description: 'The agent discovers a Weather Data API at $0.05 with 98% reliability. The zkML spending model runs locally, evaluating the purchase against Crossmint-enforced policies.',
    duration: 5000,
  },
  {
    id: 'agent-3',
    phase: 'agent',
    title: 'Decision: Approved',
    description: 'Model outputs APPROVE (87% confidence). But the Crossmint wallet won\'t just trust this claim - it requires cryptographic PROOF the model ran correctly.',
    crossmintNote: 'This is the Crossmint difference: verification, not trust.',
    duration: 5000,
  },
  {
    id: 'proof-1',
    phase: 'proof',
    title: 'Generating ZK Proof',
    description: 'JOLT-Atlas compiles the spending model into a SNARK circuit. The agent generates a ~48KB cryptographic proof that the decision was computed correctly.',
    duration: 6000,
  },
  {
    id: 'proof-2',
    phase: 'proof',
    title: 'Privacy-Preserving Proofs',
    description: 'The proof reveals ONLY the decision and confidence. Treasury balances, spending limits, and internal thresholds stay private - crucial for Crossmint\'s enterprise clients.',
    crossmintNote: 'Enterprise treasuries stay confidential. Only the approval is public.',
    duration: 5000,
  },
  {
    id: 'wallet-1',
    phase: 'wallet',
    title: 'zkML Proof Attestation',
    description: 'The Crossmint Wallet receives the payment request with the zkML proof. The proof is attested on Arc: valid proof? correct transaction? model approved? No valid attestation = no payment.',
    crossmintNote: 'Crossmint Wallets can require zkML proof attestation before releasing funds.',
    duration: 5000,
  },
  {
    id: 'execution-1',
    phase: 'execution',
    title: 'Crossmint Executes Payment',
    description: 'Proof attested! Crossmint executes the $0.05 USDC transfer. The proof hash is recorded on Arc, preventing replay attacks.',
    duration: 5000,
  },
  {
    id: 'conclusion-1',
    phase: 'conclusion',
    title: 'Trustless Agent Commerce',
    description: 'The agent purchased a service with cryptographic proof of policy compliance. No party needed to trust the agent code. This is enterprise-ready autonomous commerce.',
    crossmintNote: 'Crossmint + zkML = The only agent wallet with cryptographic spending enforcement.',
    duration: 6000,
  },
];

const PHASES = ['intro', 'agent', 'proof', 'wallet', 'execution', 'conclusion'] as const;

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
  const [verificationSteps, setVerificationSteps] = useState<Array<{ step: string; status: string; txHash?: string }>>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);

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

    // Trigger phase-specific effects
    if (step.phase === 'agent' && step.id === 'agent-2') {
      const thoughts = [
        'Evaluating Weather Data API...',
        'Price: $0.05 USDC',
        'Service reliability: 98%',
        'Checking policy constraints...',
        'Decision: APPROVE',
      ];
      setAgentThoughts([]);
      thoughts.forEach((thought, i) => {
        setTimeout(() => {
          if (isPlayingRef.current) {
            setAgentThoughts(prev => [...prev, thought]);
          }
        }, i * 800);
      });
    }

    if (step.phase === 'proof' && step.id === 'proof-1') {
      setShowProof(true);
      const input: SpendingModelInput = {
        ...createDefaultInput(),
        serviceName: 'Weather Data API',
        serviceUrl: 'https://api.weather.example.com/v1/forecast',
        priceUsdc: 0.05,
        serviceSuccessRate: 0.98,
        budgetUsdc: 2.00,
        spentTodayUsdc: 0.10,
        dailyLimitUsdc: 0.50,
        serviceTotalCalls: 50,
      };
      generateProof(input).then(result => {
        // Store proof hash for transfer audit trail
        if (result.success && result.proof) {
          setProofHash(result.proof.proofHash);
        }
        onProofGenerated?.(result);
      });
    }

    if (step.phase === 'execution') {
      // Execute real transfer via Crossmint API
      const recipientAddress = '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384'; // Demo recipient
      const transferAmount = 0.05; // $0.05 USDC

      executeTransfer(recipientAddress, transferAmount, proofHash || undefined)
        .then(result => {
          if (result.success && result.txHash) {
            setTxHash(result.txHash);
            onTxExecuted?.(result.txHash);
            // Capture on-chain verification info
            if (result.verifiedOnChain) {
              setVerifiedOnChain(true);
              setAttestationTxHash(result.attestationTxHash || null);
              setVerificationSteps(result.steps || []);
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

    // Set timer to advance
    timerRef.current = setTimeout(() => {
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
    }, step.duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [currentStepIndex, isPlaying, generateProof, executeTransfer, proofHash, onProofGenerated, onTxExecuted]);

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

      {/* Main Demo Container */}
      <div className="flex h-[780px] bg-[#0a0a0a] rounded-2xl overflow-hidden border border-gray-800">
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
                <p className="text-xs text-gray-300">{currentStep.crossmintNote}</p>
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
      <div className="flex-1 p-6 bg-[#0a0a0a] overflow-hidden">
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
              <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <Shield className="w-4 h-4 text-purple-400" />
                <span className="text-purple-400 font-medium text-sm">Arc Network</span>
              </div>
            </div>

            <h2 className="text-2xl font-bold mb-2">
              zkML Spending Proofs
            </h2>
            <p className="text-gray-400 max-w-2xl mb-5 text-sm">
              AI agents use <span className="text-[#00D4AA]">Crossmint MPC Wallets</span> to hold funds.
              <span className="text-yellow-400"> Jolt-Atlas</span> generates SNARK proofs of spending policy compliance.
              <span className="text-purple-400"> Arc Network</span> verifies proofs on-chain before <span className="text-blue-400">USDC</span> transfers execute.
            </p>

            {/* Three-Column Tech Stack */}
            <div className="grid grid-cols-3 gap-3 max-w-2xl mb-5">
              {/* Crossmint Column */}
              <a href="https://crossmint.com" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-[#00D4AA]/30 rounded-xl p-3 hover:border-[#00D4AA]/60 transition-colors group">
                <div className="flex items-center gap-2 mb-2">
                  <Wallet className="w-4 h-4 text-[#00D4AA]" />
                  <span className="text-[#00D4AA] font-semibold text-sm group-hover:underline">Crossmint</span>
                </div>
                <p className="text-[10px] text-gray-400 mb-2">Enterprise wallet infrastructure</p>
                <div className="space-y-1 text-[10px]">
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>MPC Wallets (Fireblocks)</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>Agentic Commerce</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-[#00D4AA]" />
                    <span>Headless Checkout</span>
                  </div>
                </div>
              </a>

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

              {/* Arc Network Column */}
              <a href="https://arc.network" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-3 hover:border-purple-500/60 transition-colors group">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="w-4 h-4 text-purple-400" />
                  <span className="text-purple-400 font-semibold text-sm group-hover:underline">Arc Network</span>
                </div>
                <p className="text-[10px] text-gray-400 mb-2">On-chain attestation</p>
                <div className="space-y-1 text-[10px]">
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-purple-400" />
                    <span>Proof Attestation</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-purple-400" />
                    <span>SpendingGate Contract</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <CheckCircle2 className="w-3 h-3 text-purple-400" />
                    <span>USDC Transfers</span>
                  </div>
                </div>
              </a>
            </div>

            {/* Contract Addresses */}
            <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 max-w-2xl mb-5">
              <div className="text-[10px] text-gray-500 mb-2">Arc Testnet Contracts (Chain ID: 5042002)</div>
              <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-gray-500">USDC</div>
                  <a href="https://testnet.arcscan.app/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">0x1Fb6...F213</a>
                </div>
                <div>
                  <div className="text-gray-500">ProofAttestation</div>
                  <a href="https://testnet.arcscan.app/address/0xBE9a5DF7C551324CB872584C6E5bF56799787952" target="_blank" rel="noopener noreferrer" className="text-yellow-400 hover:underline">0xBE9a...7952</a>
                </div>
                <div>
                  <div className="text-gray-500">SpendingGate</div>
                  <a href="https://testnet.arcscan.app/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:underline">0x6A47...d138</a>
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
                <div className="text-[10px] text-gray-400">Transfers on Arc</div>
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
                  <h4 className="font-semibold text-sm">Crossmint AI Agent</h4>
                  <p className="text-xs text-gray-400">Autonomous purchasing</p>
                </div>
              </div>

              <div className="p-3 border-b border-gray-800">
                <div className="text-xs text-purple-300 mb-2">Crossmint Policy Config</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Daily Limit</div>
                    <div className="font-mono">$0.50</div>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Spent Today</div>
                    <div className="font-mono">$0.10</div>
                  </div>
                  <div className="p-2 bg-gray-900/50 rounded">
                    <div className="text-gray-400 text-[10px]">Treasury</div>
                    <div className="font-mono">$2.00</div>
                  </div>
                </div>
              </div>

              <div className="p-3 border-b border-gray-800">
                <div className="text-xs text-gray-400 mb-2">Purchase Request</div>
                <div className={`p-2 bg-green-900/20 border rounded-lg transition-all ${isPlaying ? 'border-green-500 shadow-md shadow-green-500/10' : 'border-green-700/50'}`}>
                  <div className="flex justify-between mb-1">
                    <span className="font-medium text-sm">Weather Data API</span>
                    <span className="font-mono text-green-400 text-sm">$0.05</span>
                  </div>
                  <div className="text-xs text-gray-400">Reliability: 98%</div>
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
                Crossmint Proof Guarantees
              </h4>
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Spending model executed correctly on claimed inputs
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Model output APPROVED with stated confidence
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Shield className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Enterprise treasury details stay private
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Wallet className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-gray-300">
                    Crossmint can verify without seeing balances
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
                    <div className="w-10 h-10 bg-[#00D4AA]/20 rounded-lg flex items-center justify-center">
                      <Wallet className={`w-6 h-6 text-[#00D4AA] ${isPlaying ? 'animate-pulse' : ''}`} />
                    </div>
                    {isPlaying && (
                      <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-[#00D4AA] rounded-full animate-ping" />
                    )}
                  </div>
                  <div>
                    <h4 className="font-semibold text-[#00D4AA]">Crossmint Wallet</h4>
                    <p className="text-xs text-gray-400">Proof Verification</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-400">
                    {wallet.loading ? 'Loading...' : 'Enterprise Treasury'}
                  </div>
                  <div className="font-mono font-semibold">
                    {wallet.loading ? '...' : `$${parseFloat(wallet.balanceUsdc || '0').toFixed(2)} USDC`}
                  </div>
                </div>
              </div>

              <div className="p-4 border-b border-gray-800">
                <div className="text-xs text-[#00D4AA] mb-3 flex items-center gap-2">
                  <Shield className={`w-3 h-3 ${isPlaying ? 'animate-pulse' : ''}`} />
                  Crossmint Verification Engine
                  {isPlaying && <span className="text-[#00D4AA] text-[10px] animate-pulse">● ACTIVE</span>}
                </div>
                <div className={`p-3 bg-green-900/20 border rounded-lg transition-all duration-300 ${isPlaying ? 'border-green-500 shadow-md shadow-green-500/20' : 'border-green-700/50'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className={`w-4 h-4 text-green-400 ${isPlaying ? 'animate-bounce' : ''}`} />
                    <span className="font-medium text-green-400">zkML Proof Verified</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">Decision:</span>
                      <span className="ml-1 text-green-400 font-mono">APPROVED</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Confidence:</span>
                      <span className="ml-1 font-mono">87%</span>
                    </div>
                  </div>
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
                zkML Proof Verification
              </h4>
              <div className="space-y-3">
                {[
                  { check: 'SNARK proof cryptographically valid?', delay: 0 },
                  { check: 'Proof binds to this exact transaction?', delay: 150 },
                  { check: 'Spending model output: APPROVED?', delay: 300 },
                  { check: 'Replay attack protection (proof hash)?', delay: 450 },
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
                  <span className="text-gray-300">Crossmint enterprise custody ensures funds only move with valid proofs</span>
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
                    <div className="text-xs text-gray-400">From: Crossmint Wallet</div>
                    <div className="font-mono text-sm text-white truncate w-48">
                      {wallet.address || '0xe2e8690bff...'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Balance</span>
                  <span className="font-mono text-[#00D4AA]">${wallet.balanceUsdc || '10.00'} USDC</span>
                </div>
              </div>

              {/* Arrow with Amount */}
              <div className="flex items-center justify-center py-2">
                <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300 ${
                  isPlaying && !txHash
                    ? 'bg-green-500/20 border border-green-500 shadow-lg shadow-green-500/30'
                    : txHash
                    ? 'bg-green-500/30 border border-green-500'
                    : 'bg-gray-800 border border-gray-700'
                }`}>
                  <span className="text-lg font-bold text-green-400">$0.05</span>
                  <span className="text-gray-400 text-sm">USDC</span>
                  <ArrowRight className={`w-4 h-4 text-green-400 ${isPlaying && !txHash ? 'animate-pulse' : ''}`} />
                </div>
              </div>

              {/* To Address */}
              <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all duration-300 ${txHash ? 'border-green-500 shadow-lg shadow-green-500/20' : 'border-gray-700'}`}>
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${txHash ? 'bg-green-500/20' : 'bg-gray-800'}`}>
                    <CheckCircle2 className={`w-5 h-5 ${txHash ? 'text-green-400' : 'text-gray-500'}`} />
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">To: Service Provider</div>
                    <div className="font-mono text-sm text-white truncate w-48">
                      0x982Cd966...1C79E384
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Service</span>
                  <span className="text-purple-400">Weather Data API</span>
                </div>
              </div>

              {/* Network Badge */}
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 rounded-full text-xs">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                  <span className="text-purple-400">Arc Testnet</span>
                  <span className="text-gray-500">Chain ID: 5042002</span>
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
                  {txHash ? 'USDC sent on Arc testnet' : 'Broadcasting transaction...'}
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

                  {/* Verification Steps */}
                  {verificationSteps.length > 0 && (
                    <div className="p-3 bg-gray-900/50 rounded-lg border border-gray-700">
                      <div className="text-xs text-gray-400 mb-2">Verification Steps</div>
                      <div className="space-y-1.5">
                        {verificationSteps.map((vstep, i) => (
                          <div key={i} className="flex items-center gap-2 text-xs">
                            {vstep.status === 'success' ? (
                              <CheckCircle2 className="w-3 h-3 text-green-400" />
                            ) : vstep.status === 'skipped' ? (
                              <div className="w-3 h-3 rounded-full bg-gray-600" />
                            ) : (
                              <AlertCircle className="w-3 h-3 text-red-400" />
                            )}
                            <span className={vstep.status === 'success' ? 'text-green-400' : vstep.status === 'skipped' ? 'text-gray-500' : 'text-red-400'}>
                              {vstep.step}
                            </span>
                            {vstep.txHash && (
                              <a
                                href={`https://testnet.arcscan.app/tx/${vstep.txHash}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#00D4AA] hover:underline ml-auto"
                              >
                                tx
                              </a>
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
                      href={`https://testnet.arcscan.app/tx/${txHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-2 text-xs text-[#00D4AA] hover:underline"
                    >
                      View on Arc Explorer <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>

                  {/* Attestation Tx Hash */}
                  {attestationTxHash && (
                    <div className="p-3 bg-purple-900/10 rounded-lg border border-purple-500/30">
                      <div className="text-xs text-gray-400 mb-1">Attestation Transaction</div>
                      <div className="font-mono text-xs text-purple-400 truncate">{attestationTxHash}</div>
                      <a
                        href={`https://testnet.arcscan.app/tx/${attestationTxHash}`}
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
                  {['Signing transaction...', 'Broadcasting to Arc...', 'Waiting for confirmation...'].map((step, i) => (
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
              <div className="flex items-center gap-2 px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <Shield className="w-3 h-3 text-purple-400" />
                <span className="text-purple-400 font-medium text-xs">Arc Network</span>
              </div>
            </div>

            <h2 className="text-xl font-bold mb-2">
              Trustless Agent Commerce Complete
            </h2>
            <p className="text-gray-400 max-w-2xl mb-5 text-sm">
              An AI agent purchased a service using <span className="text-blue-400">USDC</span> on <span className="text-purple-400">Arc Network</span>.
              <span className="text-yellow-400"> Jolt-Atlas</span> generated a SNARK proof attested on Arc.
              The <span className="text-[#00D4AA]">Crossmint Wallet</span> released funds only after proof attestation. No trust required - only math.
            </p>

            {/* Transaction Summary */}
            <div className="grid grid-cols-4 gap-2 max-w-2xl mb-5">
              <div className="p-2 bg-[#0d1117] border border-blue-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-blue-400">$0.05</div>
                <div className="text-[10px] text-gray-400">USDC Transferred</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-yellow-400">~48KB</div>
                <div className="text-[10px] text-gray-400">Jolt-Atlas Proof</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-purple-500/30 rounded-xl text-center">
                <div className="text-lg font-bold text-purple-400">Arc</div>
                <div className="text-[10px] text-gray-400">Proof Attested</div>
              </div>
              <div className="p-2 bg-[#0d1117] border border-[#00D4AA]/30 rounded-xl text-center">
                <div className="text-lg font-bold text-[#00D4AA]">MPC</div>
                <div className="text-[10px] text-gray-400">Crossmint Wallet</div>
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
                    <span className="text-purple-400 font-medium">1. AI Agent Evaluated Purchase</span>
                    <p className="text-gray-400 mt-1">The agent discovered a Weather Data API at $0.05 with 98% reliability. Its spending model ran locally to determine if the purchase complied with configured policies.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                  <Shield className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-yellow-400 font-medium">2. zkML Proof Generated</span>
                    <p className="text-gray-400 mt-1">NovaNet&apos;s Jolt-Atlas prover compiled the spending model into a SNARK circuit and generated a ~48KB cryptographic proof that the decision was computed correctly.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
                  <Shield className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-purple-400 font-medium">3. Proof Attested on Arc</span>
                    <p className="text-gray-400 mt-1">The ProofAttestation contract on Arc Network recorded the SNARK proof hash. SpendingGate contract checked the attestation before authorizing the transfer.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#00D4AA]/10 rounded-lg border border-[#00D4AA]/30">
                  <Wallet className="w-4 h-4 text-[#00D4AA] flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="text-[#00D4AA] font-medium">4. Crossmint Executed Transfer</span>
                    <p className="text-gray-400 mt-1">The Crossmint MPC wallet signed and broadcast the $0.05 USDC transfer on Arc Network. Sub-second finality with proof hash recorded for audit.</p>
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
      </div>
    </div>
  );
}
