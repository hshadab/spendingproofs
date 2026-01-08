'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Bot,
  Zap,
  Shield,
  CheckCircle2,
  XCircle,
  Battery,
  DollarSign,
  ExternalLink,
  Package,
  Settings,
  Wifi,
  WifiOff,
  RefreshCw,
  Info,
  Sparkles,
} from 'lucide-react';
import {
  type DemoStep,
  type RobotSpendingPolicy,
  type RobotProofResult,
  SAMPLE_ROBOTS,
  SAMPLE_SERVICES,
  DEFAULT_ROBOT_POLICY,
  getCategoryDisplayName,
  generateRobotPaymentProof,
  generateMockProof,
  evaluatePaymentRequest,
} from '@/lib/openmind';
import { ADDRESSES, ARC_CHAIN } from '@/lib/config';

// Contract addresses for display
const CONTRACTS = {
  usdc: ADDRESSES.usdc,
  proofAttestation: ADDRESSES.proofAttestation,
  spendingGate: ADDRESSES.spendingGate,
} as const;

// Demo step definitions
const DEMO_STEPS: DemoStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'OpenMind + zkML Spending Proofs',
    description: 'See how autonomous robots using OpenMind\'s OM1 operating system can make USDC payments via the x402 protocol - with cryptographic proof they followed their spending policies.',
    technicalNote: 'OpenMind partnered with Circle in Dec 2025 to enable the first autonomous robot payments using USDC.',
    duration: 5000,
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'The Challenge of Robot Commerce',
    description: 'Robots operating autonomously need to pay for charging, APIs, compute, and data. But owners can\'t monitor every transaction. How do you trust a robot with your money?',
    technicalNote: 'From OpenMind docs: "autonomous actions by your robot are inherently unpredictable"',
    duration: 5000,
  },
  {
    id: 'robot-1',
    phase: 'robot',
    title: 'Meet DeliveryBot-7',
    description: 'An autonomous delivery robot running OpenMind OM1. It has a USDC wallet on Arc testnet for paying operational expenses via the x402 protocol.',
    duration: 4000,
  },
  {
    id: 'robot-2',
    phase: 'robot',
    title: 'Robot Wallet State',
    description: 'Fetching real wallet balance from Arc testnet...',
    duration: 4000,
  },
  {
    id: 'service-1',
    phase: 'service',
    title: 'Robot Needs to Charge',
    description: 'Battery at 23%. The robot discovers a nearby ChargePoint station offering fast charging for $0.50 USDC via x402.',
    duration: 4000,
  },
  {
    id: 'policy-1',
    phase: 'policy',
    title: 'Owner\'s Spending Policy',
    description: 'The robot owner configured strict limits: $10/day max, $2 per transaction, only charging/navigation/compute/data categories, minimum 85% service reliability.',
    duration: 5000,
  },
  {
    id: 'policy-2',
    phase: 'policy',
    title: 'OpenMind LLM Decision',
    description: 'Calling OpenMind API to evaluate the spending decision...',
    duration: 5000,
  },
  {
    id: 'proof-1',
    phase: 'proof',
    title: 'Generating zkML Proof',
    description: 'The x402 payment gateway requires a cryptographic PROOF that the policy model was executed correctly.',
    duration: 4000,
  },
  {
    id: 'proof-2',
    phase: 'proof',
    title: 'Jolt-Atlas SNARK',
    description: 'Generating a ~48KB zero-knowledge proof using Jolt-Atlas. This proves the spending decision was computed correctly without revealing the full policy.',
    duration: 6000,
  },
  {
    id: 'payment-1',
    phase: 'payment',
    title: 'x402 Payment with Proof',
    description: 'The robot submits the payment request with the zkML proof attached. The charging station verifies the proof before accepting payment.',
    duration: 4000,
  },
  {
    id: 'payment-2',
    phase: 'payment',
    title: 'USDC Transfer on Arc',
    description: 'USDC transferred on Arc testnet. The proof hash is stored on-chain as an audit trail. Robot begins charging.',
    duration: 4000,
  },
  {
    id: 'conclusion-1',
    phase: 'conclusion',
    title: 'Trustless Robot Commerce',
    description: 'Every robot payment is now cryptographically verified. Owners can deploy fleets of autonomous robots knowing spending policies are mathematically enforced.',
    technicalNote: 'This enables the "embodied intelligence economy" Circle and OpenMind envision.',
    duration: 5000,
  },
];

// Phase colors
const PHASE_COLORS: Record<string, string> = {
  intro: 'from-blue-500 to-cyan-500',
  robot: 'from-purple-500 to-pink-500',
  service: 'from-amber-500 to-orange-500',
  policy: 'from-green-500 to-emerald-500',
  proof: 'from-yellow-500 to-amber-500',
  payment: 'from-cyan-500 to-blue-500',
  conclusion: 'from-pink-500 to-purple-500',
};

const PHASE_LABELS: Record<string, string> = {
  intro: 'Introduction',
  robot: 'Robot Agent',
  service: 'Service Request',
  policy: 'Policy Check',
  proof: 'zkML Proof',
  payment: 'Payment',
  conclusion: 'Complete',
};

const PHASES = ['intro', 'robot', 'service', 'policy', 'proof', 'payment', 'conclusion'] as const;

interface WalletInfo {
  address: string;
  balanceUsdc: number;
  network: string;
  explorerUrl: string;
}

interface LLMDecision {
  decision: 'approve' | 'reject';
  confidence: number;
  reasoning: string;
  riskFactors: string[];
}

export function GuidedDemo() {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [useRealProver, setUseRealProver] = useState(false);
  const [useRealLLM, setUseRealLLM] = useState(true);
  const [proofResult, setProofResult] = useState<RobotProofResult | null>(null);
  const [proofProgress, setProofProgress] = useState(0);
  const [proofStatus, setProofStatus] = useState('');
  const [isGeneratingProof, setIsGeneratingProof] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);

  // Real API state
  const [walletInfo, setWalletInfo] = useState<WalletInfo | null>(null);
  const [walletLoading, setWalletLoading] = useState(false);
  const [llmDecision, setLlmDecision] = useState<LLMDecision | null>(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  // Payment state
  const [useRealPayment, setUseRealPayment] = useState(false);
  const [paymentResult, setPaymentResult] = useState<{
    txHash: string;
    explorerUrl: string;
  } | null>(null);
  const [paymentLoading, setPaymentLoading] = useState(false);

  // Demo state
  const [robot] = useState(SAMPLE_ROBOTS[0]);
  const [service] = useState(SAMPLE_SERVICES[0]); // Charging station
  const [policy] = useState<RobotSpendingPolicy>(DEFAULT_ROBOT_POLICY);
  const [spentToday] = useState(3.50);

  const currentStep = DEMO_STEPS[currentStepIndex];
  const currentPhaseIndex = PHASES.indexOf(currentStep.phase as typeof PHASES[number]);

  // Keep ref in sync with state
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Fetch real wallet balance
  const fetchWalletBalance = useCallback(async () => {
    setWalletLoading(true);
    try {
      const response = await fetch('/api/openmind/wallet');
      const data = await response.json();
      if (data.success) {
        setWalletInfo(data.wallet);
        setApiStatus('connected');
      }
    } catch (error) {
      console.error('Failed to fetch wallet:', error);
      setApiStatus('disconnected');
    } finally {
      setWalletLoading(false);
    }
  }, []);

  // Get LLM decision
  const fetchLLMDecision = useCallback(async () => {
    if (!useRealLLM) {
      // Use local evaluation
      const evaluation = evaluatePaymentRequest({
        robotId: robot.id,
        robotType: robot.type,
        service,
        walletState: {
          address: (walletInfo?.address || '0x0') as `0x${string}`,
          balanceUsdc: walletInfo?.balanceUsdc || 100,
          spentTodayUsdc: spentToday,
          txCountToday: 5,
        },
        policy,
        timestamp: Date.now(),
      });
      setLlmDecision({
        decision: evaluation.approved ? 'approve' : 'reject',
        confidence: 0.95,
        reasoning: evaluation.approved ? 'All policy checks passed' : evaluation.reasons.join(', '),
        riskFactors: evaluation.reasons,
      });
      return;
    }

    setLlmLoading(true);
    try {
      const response = await fetch('/api/openmind/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          robotId: robot.id,
          robotName: robot.name,
          currentTask: robot.currentTask,
          batteryLevel: 23,
          serviceRequest: {
            name: service.serviceName,
            category: service.category,
            price: service.priceUsdc,
            reliability: service.reliabilityScore,
            description: service.description,
          },
          policy: {
            dailyLimit: policy.dailyLimitUsdc,
            maxSingleTx: policy.maxSingleTxUsdc,
            allowedCategories: policy.allowedCategories,
            minReliability: policy.minServiceReliability,
          },
          spentToday,
        }),
      });
      const data = await response.json();
      if (data.success) {
        setLlmDecision(data.decision);
      }
    } catch (error) {
      console.error('Failed to get LLM decision:', error);
    } finally {
      setLlmLoading(false);
    }
  }, [robot, service, policy, walletInfo, spentToday, useRealLLM]);

  // Initial fetch
  useEffect(() => {
    fetchWalletBalance();
  }, [fetchWalletBalance]);

  // Auto-advance when playing
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    // Don't auto-advance during proof generation
    if (currentStep.phase === 'proof' && currentStep.id === 'proof-2' && !proofResult) {
      return;
    }

    timerRef.current = setTimeout(() => {
      if (isPlayingRef.current) {
        if (currentStepIndex < DEMO_STEPS.length - 1) {
          setCurrentStepIndex((i) => i + 1);
        } else {
          setIsPlaying(false);
        }
      }
    }, currentStep.duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [isPlaying, currentStepIndex, currentStep, proofResult]);

  // Trigger LLM decision at policy-2 step
  useEffect(() => {
    if (currentStep.id === 'policy-2' && !llmDecision && !llmLoading) {
      fetchLLMDecision();
    }
  }, [currentStep.id, llmDecision, llmLoading, fetchLLMDecision]);

  // Trigger proof generation at the right step
  useEffect(() => {
    if (currentStep.id === 'proof-2' && !proofResult && !isGeneratingProof) {
      generateProof();
    }
  }, [currentStep.id, proofResult, isGeneratingProof]);

  const generateProof = useCallback(async () => {
    setIsGeneratingProof(true);
    setProofProgress(0);
    setProofStatus('Starting...');

    const request = {
      robotId: robot.id,
      robotType: robot.type,
      service,
      walletState: {
        address: (walletInfo?.address || '0x0') as `0x${string}`,
        balanceUsdc: walletInfo?.balanceUsdc || 100,
        spentTodayUsdc: spentToday,
        txCountToday: 5,
      },
      policy,
      timestamp: Date.now(),
    };

    try {
      const prover = useRealProver ? generateRobotPaymentProof : generateMockProof;
      const result = await prover(request, (progress, status) => {
        setProofProgress(progress);
        setProofStatus(status);
      });
      setProofResult(result);
    } catch (error) {
      console.error('Proof generation failed:', error);
      setProofStatus('Error generating proof');
    } finally {
      setIsGeneratingProof(false);
    }
  }, [robot, service, walletInfo, spentToday, policy, useRealProver]);

  // Execute real payment
  const executePayment = useCallback(async () => {
    if (!useRealPayment || paymentResult || paymentLoading) return;

    setPaymentLoading(true);
    try {
      const response = await fetch('/api/openmind/transfer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to: service.providerAddress,
          amount: service.priceUsdc,
          proofHash: proofResult?.proofHash,
        }),
      });
      const data = await response.json();
      if (data.success) {
        setPaymentResult({
          txHash: data.transfer.txHash,
          explorerUrl: data.transfer.explorerUrl,
        });
        // Refresh wallet balance after payment
        fetchWalletBalance();
      }
    } catch (error) {
      console.error('Payment failed:', error);
    } finally {
      setPaymentLoading(false);
    }
  }, [useRealPayment, paymentResult, paymentLoading, service, proofResult, fetchWalletBalance]);

  // Trigger payment execution when reaching payment phase with real payment enabled
  useEffect(() => {
    if (currentStep.phase === 'payment' && currentStep.id === 'payment-2' &&
        proofResult?.approved && useRealPayment && !paymentResult && !paymentLoading) {
      executePayment();
    }
  }, [currentStep, proofResult, useRealPayment, paymentResult, paymentLoading, executePayment]);

  const handleReset = () => {
    setCurrentStepIndex(0);
    setIsPlaying(false);
    setProofResult(null);
    setProofProgress(0);
    setProofStatus('');
    setLlmDecision(null);
    setPaymentResult(null);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
  };

  const handleNext = () => {
    if (currentStepIndex < DEMO_STEPS.length - 1) {
      setCurrentStepIndex((i) => i + 1);
    }
  };

  const handlePrev = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex((i) => i - 1);
      if (currentStepIndex <= DEMO_STEPS.findIndex((s) => s.id === 'proof-2')) {
        setProofResult(null);
      }
      if (currentStepIndex <= DEMO_STEPS.findIndex((s) => s.id === 'policy-2')) {
        setLlmDecision(null);
      }
    }
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'intro': return <Info className="w-4 h-4" />;
      case 'robot': return <Bot className="w-4 h-4" />;
      case 'service': return <Zap className="w-4 h-4" />;
      case 'policy': return <Shield className="w-4 h-4" />;
      case 'proof': return <Sparkles className="w-4 h-4" />;
      case 'payment': return <DollarSign className="w-4 h-4" />;
      case 'conclusion': return <CheckCircle2 className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  return (
    <div className="flex flex-col">
      {/* Playback Controls - Separate Control Panel Above Demo */}
      <div className="mb-3 p-3 bg-[#0d1117] rounded-xl border border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handlePrev}
              disabled={currentStepIndex <= 0}
              className="p-2.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-50 border border-gray-700"
              title="Previous Step"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg transition-all text-sm font-medium ${
                isPlaying
                  ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30'
                  : 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 border border-cyan-500/30'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isPlaying ? 'Pause' : 'Play Demo'}</span>
            </button>
            <button
              type="button"
              onClick={handleNext}
              disabled={currentStepIndex >= DEMO_STEPS.length - 1}
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

          {/* Toggles */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-gray-400">LLM:</span>
              <button
                onClick={() => setUseRealLLM(!useRealLLM)}
                className={`px-2 py-1 rounded font-medium transition-colors ${
                  useRealLLM ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
                }`}
              >
                {useRealLLM ? 'OpenMind' : 'Local'}
              </button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Prover:</span>
              <button
                onClick={() => setUseRealProver(!useRealProver)}
                className={`px-2 py-1 rounded font-medium transition-colors ${
                  useRealProver ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
                }`}
              >
                {useRealProver ? 'Real' : 'Simulated'}
              </button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Payment:</span>
              <button
                onClick={() => setUseRealPayment(!useRealPayment)}
                className={`px-2 py-1 rounded font-medium transition-colors ${
                  useRealPayment ? 'bg-red-600 text-white' : 'bg-gray-700 text-gray-300'
                }`}
                title={useRealPayment ? 'Will spend real testnet USDC!' : 'Simulated payment'}
              >
                {useRealPayment ? 'Real USDC' : 'Simulated'}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-gray-400">
            <span>Step {currentStepIndex + 1}/{DEMO_STEPS.length}</span>
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
              zkML Spending Proofs for<br />
              <span className="text-cyan-400 font-medium">OpenMind Robot Payments (x402)</span>
            </p>

            {/* API Status */}
            <div className="mt-3 flex items-center gap-2 text-xs">
              {apiStatus === 'connected' ? (
                <div className="flex items-center gap-1.5 text-green-400">
                  <Wifi className="w-3 h-3" />
                  <span>Arc Testnet</span>
                </div>
              ) : apiStatus === 'checking' ? (
                <div className="flex items-center gap-1.5 text-yellow-400">
                  <RefreshCw className="w-3 h-3 animate-spin" />
                  <span>Connecting...</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-red-400">
                  <WifiOff className="w-3 h-3" />
                  <span>Disconnected</span>
                </div>
              )}
              {walletInfo && (
                <span className="text-gray-500 ml-auto font-mono">
                  ${walletInfo.balanceUsdc.toFixed(2)}
                </span>
              )}
            </div>
          </div>

          {/* Current Step Annotation */}
          <div className="p-4 overflow-y-auto flex-1">
            {/* Step Badge */}
            <div className="mb-2">
              <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${PHASE_COLORS[currentStep.phase]} text-white`}>
                {PHASE_LABELS[currentStep.phase]}
              </span>
            </div>

            {/* Step Title & Description */}
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <h3 className="text-lg font-bold text-white mb-2">{currentStep.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed mb-3">{currentStep.description}</p>
              </motion.div>
            </AnimatePresence>

            {/* Technical Note */}
            {currentStep.technicalNote && (
              <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg mb-3">
                <div className="flex items-start gap-2">
                  <Bot className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-gray-300">{currentStep.technicalNote}</p>
                </div>
              </div>
            )}

            {/* Progress Bar */}
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
                          ? `bg-gradient-to-r ${PHASE_COLORS[phase]} ${isPlaying ? 'animate-pulse' : ''}`
                          : 'bg-gray-700'
                      }`}
                    />
                  );
                })}
              </div>
            </div>

            {/* Phase List */}
            <div className="grid grid-cols-2 gap-1.5 mt-4">
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
                        ? `bg-gradient-to-r ${PHASE_COLORS[phase]} text-white`
                        : 'bg-gray-700 text-gray-400'
                    }`}>
                      {isComplete ? <CheckCircle2 className="w-3 h-3" /> : getPhaseIcon(phase)}
                    </div>
                    <span className={`text-xs ${
                      isCurrent ? 'text-white font-medium' : isComplete ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {PHASE_LABELS[phase]}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Contracts Section */}
          <div className="px-4 py-2 border-t border-gray-800">
            <div className="text-[10px] text-gray-500 mb-1.5">Smart Contracts (Arc Testnet)</div>
            <div className="space-y-1">
              <a
                href={`${ARC_CHAIN.explorerUrl}/address/${CONTRACTS.spendingGate}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between text-[10px] hover:text-cyan-400 transition-colors group"
              >
                <span className="text-gray-400 group-hover:text-cyan-400">SpendingGate</span>
                <span className="font-mono text-gray-600 group-hover:text-cyan-400">{CONTRACTS.spendingGate.slice(0, 6)}...{CONTRACTS.spendingGate.slice(-4)}</span>
              </a>
              <a
                href={`${ARC_CHAIN.explorerUrl}/address/${CONTRACTS.proofAttestation}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between text-[10px] hover:text-yellow-400 transition-colors group"
              >
                <span className="text-gray-400 group-hover:text-yellow-400">ProofAttestation</span>
                <span className="font-mono text-gray-600 group-hover:text-yellow-400">{CONTRACTS.proofAttestation.slice(0, 6)}...{CONTRACTS.proofAttestation.slice(-4)}</span>
              </a>
              <a
                href={`${ARC_CHAIN.explorerUrl}/address/${CONTRACTS.usdc}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between text-[10px] hover:text-green-400 transition-colors group"
              >
                <span className="text-gray-400 group-hover:text-green-400">USDC Token</span>
                <span className="font-mono text-gray-600 group-hover:text-green-400">{CONTRACTS.usdc.slice(0, 6)}...{CONTRACTS.usdc.slice(-4)}</span>
              </a>
            </div>
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-800">
            <div className="flex items-center justify-between text-[10px] text-gray-500">
              <a href="https://openmind.org" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-cyan-400 transition-colors">
                OpenMind <ExternalLink className="w-2.5 h-2.5" />
              </a>
              <span>Powered by zkML</span>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 p-6 bg-[#0a0a0a] overflow-y-auto">
          <AnimatePresence mode="wait">
            {/* Intro Phase */}
            {currentStep.phase === 'intro' && (
              <motion.div
                key="intro"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* Powered By Header */}
                <div className="flex items-center gap-4 mb-4">
                  <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                    <Bot className="w-4 h-4 text-cyan-400" />
                    <span className="text-cyan-400 font-medium text-sm">OpenMind</span>
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
                  Robot Spending Proofs
                </h2>
                <p className="text-gray-400 max-w-2xl mb-5 text-sm">
                  Autonomous robots using <span className="text-cyan-400">OpenMind OM1</span> can pay for services via <span className="text-blue-400">x402</span>.
                  <span className="text-yellow-400"> Jolt-Atlas</span> generates SNARK proofs of spending policy compliance.
                  Payments execute on <span className="text-purple-400">Arc Network</span> with <span className="text-green-400">USDC</span>.
                </p>

                {/* Three-Column Tech Stack */}
                <div className="grid grid-cols-3 gap-3 max-w-2xl mb-5">
                  <a href="https://openmind.org" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-3 hover:border-cyan-500/60 transition-colors group">
                    <div className="flex items-center gap-2 mb-2">
                      <Bot className="w-4 h-4 text-cyan-400" />
                      <span className="text-cyan-400 font-semibold text-sm group-hover:underline">OpenMind</span>
                    </div>
                    <p className="text-[10px] text-gray-400 mb-2">Robot operating system</p>
                    <div className="space-y-1 text-[10px]">
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                        <span>OM1 Robot OS</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                        <span>x402 Payments</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                        <span>LLM Decision Engine</span>
                      </div>
                    </div>
                  </a>

                  <a href="https://github.com/ICME-Lab/jolt-atlas" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-3 hover:border-yellow-500/60 transition-colors group">
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

                  <a href="https://arc.network" target="_blank" rel="noopener noreferrer" className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-3 hover:border-purple-500/60 transition-colors group">
                    <div className="flex items-center gap-2 mb-2">
                      <Shield className="w-4 h-4 text-purple-400" />
                      <span className="text-purple-400 font-semibold text-sm group-hover:underline">Arc Network</span>
                    </div>
                    <p className="text-[10px] text-gray-400 mb-2">On-chain settlement</p>
                    <div className="space-y-1 text-[10px]">
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-purple-400" />
                        <span>USDC Transfers</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-purple-400" />
                        <span>Proof Attestation</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-500">
                        <CheckCircle2 className="w-3 h-3 text-purple-400" />
                        <span>SpendingGate</span>
                      </div>
                    </div>
                  </a>
                </div>

                {/* Workflow Overview */}
                <div className="grid grid-cols-4 gap-3 max-w-2xl">
                  <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 text-center ${isPlaying ? 'border-purple-500/50 shadow-lg shadow-purple-500/10' : 'border-gray-800'}`}>
                    <Bot className={`w-6 h-6 text-purple-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                    <div className="font-medium text-sm mb-1">Robot Agent</div>
                    <div className="text-[10px] text-gray-400">Needs service</div>
                  </div>
                  <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-100 text-center ${isPlaying ? 'border-green-500/50 shadow-lg shadow-green-500/10' : 'border-gray-800'}`}>
                    <Shield className={`w-6 h-6 text-green-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                    <div className="font-medium text-sm mb-1">Policy Check</div>
                    <div className="text-[10px] text-gray-400">LLM evaluates</div>
                  </div>
                  <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-200 text-center ${isPlaying ? 'border-yellow-500/50 shadow-lg shadow-yellow-500/10' : 'border-gray-800'}`}>
                    <Zap className={`w-6 h-6 text-yellow-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                    <div className="font-medium text-sm mb-1">zkML Proof</div>
                    <div className="text-[10px] text-gray-400">Proves compliance</div>
                  </div>
                  <div className={`p-3 bg-[#0d1117] border rounded-xl transition-all duration-500 delay-300 text-center ${isPlaying ? 'border-blue-500/50 shadow-lg shadow-blue-500/10' : 'border-gray-800'}`}>
                    <DollarSign className={`w-6 h-6 text-blue-400 mx-auto mb-2 ${isPlaying ? 'animate-pulse' : ''}`} />
                    <div className="font-medium text-sm mb-1">x402 Payment</div>
                    <div className="text-[10px] text-gray-400">USDC on Arc</div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Robot Phase */}
            {currentStep.phase === 'robot' && (
              <motion.div
                key="robot"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-4"
              >
                {/* Robot Agent Card */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-purple-500 shadow-lg shadow-purple-500/20' : 'border-purple-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center gap-3 bg-purple-500/10">
                    <div className="relative">
                      <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <Bot className="w-6 h-6 text-purple-400" />
                      </div>
                      {isPlaying && (
                        <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse" />
                      )}
                    </div>
                    <div>
                      <h4 className="font-semibold">{robot.name}</h4>
                      <p className="text-xs text-gray-400">OpenMind OM1 • {robot.location}</p>
                    </div>
                  </div>
                  <div className="p-4 space-y-3">
                    <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                      <Package className="w-5 h-5 text-blue-400" />
                      <div>
                        <div className="text-xs text-gray-400">Current Task</div>
                        <div className="text-sm">{robot.currentTask}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                      <Battery className="w-5 h-5 text-amber-400" />
                      <div>
                        <div className="text-xs text-gray-400">Battery Level</div>
                        <div className="text-sm text-amber-400">23% - Needs Charging</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Wallet Card */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-green-500 shadow-lg shadow-green-500/20' : 'border-green-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-green-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                        <DollarSign className="w-5 h-5 text-green-400" />
                      </div>
                      <div>
                        <h4 className="font-semibold">USDC Wallet</h4>
                        <p className="text-xs text-gray-400">Arc Testnet</p>
                      </div>
                    </div>
                    <span className="text-xs bg-green-600 px-1.5 py-0.5 rounded">REAL</span>
                  </div>
                  <div className="p-4 space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-xs text-gray-400 mb-1">Balance</div>
                        <div className="text-xl font-bold text-green-400">
                          {walletLoading ? '...' : `$${(walletInfo?.balanceUsdc || 0).toFixed(2)}`}
                        </div>
                      </div>
                      <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-xs text-gray-400 mb-1">Spent Today</div>
                        <div className="text-xl font-bold">${spentToday.toFixed(2)}</div>
                      </div>
                    </div>
                    {walletInfo && (
                      <a
                        href={walletInfo.explorerUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-cyan-400 hover:text-cyan-300 font-mono truncate flex items-center gap-1"
                      >
                        {walletInfo.address.slice(0, 10)}...{walletInfo.address.slice(-8)}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    <button
                      onClick={fetchWalletBalance}
                      disabled={walletLoading}
                      className="w-full text-xs text-gray-400 hover:text-white flex items-center justify-center gap-1 py-2 border border-gray-700 rounded-lg hover:bg-gray-800 transition-colors"
                    >
                      <RefreshCw className={`w-3 h-3 ${walletLoading ? 'animate-spin' : ''}`} />
                      Refresh Balance
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Service Phase */}
            {currentStep.phase === 'service' && (
              <motion.div
                key="service"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-4"
              >
                {/* Service Request */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-amber-500 shadow-lg shadow-amber-500/20' : 'border-amber-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center gap-3 bg-amber-500/10">
                    <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center">
                      <Zap className={`w-5 h-5 text-amber-400 ${isPlaying ? 'animate-pulse' : ''}`} />
                    </div>
                    <div>
                      <h4 className="font-semibold">Service Discovery</h4>
                      <p className="text-xs text-gray-400">x402 Payment Request</p>
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <div className="font-semibold">{service.serviceName}</div>
                          <div className="text-sm text-gray-400">{service.description}</div>
                        </div>
                        <div className="text-2xl font-bold text-amber-400">${service.priceUsdc.toFixed(2)}</div>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <span className="px-2 py-1 bg-gray-800 rounded">{getCategoryDisplayName(service.category)}</span>
                        <span className="text-green-400">{(service.reliabilityScore * 100).toFixed(0)}% reliability</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Robot Status */}
                <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4">
                  <h4 className="font-semibold mb-4 flex items-center gap-2">
                    <Bot className="w-5 h-5 text-purple-400" />
                    Robot Status
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                      <Battery className="w-5 h-5 text-red-400" />
                      <div className="flex-1">
                        <div className="text-sm text-red-400">Low Battery Warning</div>
                        <div className="h-2 bg-gray-800 rounded-full mt-1 overflow-hidden">
                          <div className="h-full bg-red-500 w-[23%]" />
                        </div>
                      </div>
                      <span className="text-red-400 font-mono">23%</span>
                    </div>
                    <div className="text-sm text-gray-400">
                      Robot needs to charge to continue deliveries. Found nearby ChargePoint station.
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Policy Phase */}
            {currentStep.phase === 'policy' && (
              <motion.div
                key="policy"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-4"
              >
                {/* Policy Config */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-green-500 shadow-lg shadow-green-500/20' : 'border-green-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center gap-3 bg-green-500/10">
                    <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                      <Settings className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <h4 className="font-semibold">Spending Policy</h4>
                      <p className="text-xs text-gray-400">Owner-configured limits</p>
                    </div>
                  </div>
                  <div className="p-4 space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-xs text-gray-400 mb-1">Daily Limit</div>
                        <div className="text-lg font-bold">${policy.dailyLimitUsdc.toFixed(2)}</div>
                      </div>
                      <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-xs text-gray-400 mb-1">Max per Tx</div>
                        <div className="text-lg font-bold">${policy.maxSingleTxUsdc.toFixed(2)}</div>
                      </div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Min Reliability</div>
                      <div className="font-mono">{(policy.minServiceReliability * 100).toFixed(0)}%</div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                      <div className="text-xs text-gray-400 mb-1">Allowed Categories</div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {policy.allowedCategories.map((cat) => (
                          <span key={cat} className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded">
                            {getCategoryDisplayName(cat)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* LLM Decision */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying ? 'border-cyan-500 shadow-lg shadow-cyan-500/20' : 'border-cyan-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-cyan-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                        <Bot className={`w-5 h-5 text-cyan-400 ${llmLoading ? 'animate-pulse' : ''}`} />
                      </div>
                      <div>
                        <h4 className="font-semibold">LLM Decision</h4>
                        <p className="text-xs text-gray-400">{useRealLLM ? 'OpenMind API' : 'Local Evaluation'}</p>
                      </div>
                    </div>
                    {useRealLLM && <span className="text-xs bg-green-600 px-1.5 py-0.5 rounded">REAL</span>}
                  </div>
                  <div className="p-4">
                    {llmLoading ? (
                      <div className="flex items-center gap-2 text-gray-400 py-8 justify-center">
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        <span>Calling OpenMind API...</span>
                      </div>
                    ) : llmDecision ? (
                      <div className="space-y-3">
                        <div className={`flex items-center gap-3 p-4 rounded-lg ${
                          llmDecision.decision === 'approve' ? 'bg-green-900/20 border border-green-800' : 'bg-red-900/20 border border-red-800'
                        }`}>
                          {llmDecision.decision === 'approve' ? (
                            <CheckCircle2 className="w-8 h-8 text-green-400" />
                          ) : (
                            <XCircle className="w-8 h-8 text-red-400" />
                          )}
                          <div>
                            <div className={`text-xl font-bold ${llmDecision.decision === 'approve' ? 'text-green-400' : 'text-red-400'}`}>
                              {llmDecision.decision.toUpperCase()}
                            </div>
                            <div className="text-sm text-gray-400">
                              Confidence: {(llmDecision.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-xs text-gray-400 mb-1">Reasoning</div>
                          <div className="text-sm">{llmDecision.reasoning}</div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        Waiting for decision...
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Proof Phase */}
            {currentStep.phase === 'proof' && (
              <motion.div
                key="proof"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-4"
              >
                {/* Proof Generation */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all duration-300 ${isPlaying && isGeneratingProof ? 'border-yellow-500 shadow-lg shadow-yellow-500/20' : 'border-yellow-500/50'}`}>
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-yellow-500/10">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                        <Zap className={`w-5 h-5 text-yellow-400 ${isGeneratingProof ? 'animate-pulse' : ''}`} />
                      </div>
                      <div>
                        <h4 className="font-semibold">zkML Proof</h4>
                        <p className="text-xs text-gray-400">Jolt-Atlas SNARK</p>
                      </div>
                    </div>
                    {useRealProver && <span className="text-xs bg-yellow-600 px-1.5 py-0.5 rounded">REAL</span>}
                  </div>
                  <div className="p-4">
                    {isGeneratingProof ? (
                      <div className="space-y-4">
                        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-gradient-to-r from-yellow-500 to-amber-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${proofProgress}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">{proofStatus}</span>
                          <span className="text-yellow-400 font-bold">{proofProgress}%</span>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500">
                          {['Preparing', 'Generating', 'Signing', 'Verifying'].map((stage, i) => {
                            const thresholds = [20, 70, 85, 100];
                            const isComplete = proofProgress >= thresholds[i];
                            const isActive = proofProgress >= (i === 0 ? 0 : thresholds[i-1]) && proofProgress < thresholds[i];
                            return (
                              <span key={stage} className={isComplete ? 'text-green-400' : isActive ? 'text-yellow-400' : ''}>
                                {isComplete ? '✓' : isActive ? '●' : '○'} {stage}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    ) : proofResult ? (
                      <div className="space-y-4">
                        <div className={`flex items-center gap-3 p-4 rounded-lg ${
                          proofResult.approved ? 'bg-green-900/20 border border-green-800' : 'bg-red-900/20 border border-red-800'
                        }`}>
                          {proofResult.approved ? (
                            <CheckCircle2 className="w-8 h-8 text-green-400" />
                          ) : (
                            <XCircle className="w-8 h-8 text-red-400" />
                          )}
                          <div>
                            <div className={`text-lg font-bold ${proofResult.approved ? 'text-green-400' : 'text-red-400'}`}>
                              Proof {proofResult.approved ? 'Verified' : 'Failed'}
                            </div>
                            <div className="text-sm text-gray-400">
                              Confidence: {(proofResult.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="text-xs text-gray-400 mb-1">Proof Size</div>
                            <div className="font-mono">{(proofResult.proofSizeBytes / 1000).toFixed(0)} KB</div>
                          </div>
                          <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="text-xs text-gray-400 mb-1">Generation</div>
                            <div className="font-mono">{(proofResult.generationTimeMs / 1000).toFixed(1)}s</div>
                          </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                          <div className="text-xs text-gray-400 mb-1">Proof Hash</div>
                          <div className="font-mono text-xs text-cyan-400 truncate">{proofResult.proofHash}</div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        Waiting for proof generation...
                      </div>
                    )}
                  </div>
                </div>

                {/* Proof Guarantees */}
                <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4">
                  <h4 className="font-semibold mb-4 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-yellow-400" />
                    Proof Guarantees
                  </h4>
                  <div className="space-y-3">
                    {[
                      'Spending model executed correctly',
                      'Policy constraints were checked',
                      'Transaction bound to this request',
                      'No policy parameters exposed',
                    ].map((guarantee, i) => (
                      <div key={i} className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                        proofResult ? 'bg-green-900/20 border border-green-500/30' : 'bg-gray-800/50'
                      }`}>
                        <CheckCircle2 className={`w-5 h-5 ${proofResult ? 'text-green-400' : 'text-gray-500'}`} />
                        <span className="text-sm">{guarantee}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Payment Phase */}
            {currentStep.phase === 'payment' && (
              <motion.div
                key="payment"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-2 gap-4"
              >
                {/* Payment Flow */}
                <div className="space-y-3">
                  {/* From */}
                  <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all ${isPlaying ? 'border-purple-500' : 'border-gray-700'}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <Bot className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <div className="text-xs text-gray-400">From: Robot Wallet</div>
                        <div className="font-mono text-sm truncate">
                          {walletInfo?.address.slice(0, 10)}...{walletInfo?.address.slice(-8)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Amount */}
                  <div className="flex items-center justify-center py-2">
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${
                      isPlaying ? 'bg-green-500/20 border border-green-500' : 'bg-gray-800 border border-gray-700'
                    }`}>
                      <span className="text-lg font-bold text-green-400">${service.priceUsdc.toFixed(2)}</span>
                      <span className="text-gray-400 text-sm">USDC</span>
                      <DollarSign className={`w-4 h-4 text-green-400 ${isPlaying ? 'animate-bounce' : ''}`} />
                    </div>
                  </div>

                  {/* To */}
                  <div className={`bg-[#0d1117] border rounded-xl p-4 transition-all ${isPlaying ? 'border-amber-500' : 'border-gray-700'}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center">
                        <Zap className="w-5 h-5 text-amber-400" />
                      </div>
                      <div>
                        <div className="text-xs text-gray-400">To: {service.serviceName}</div>
                        <div className="font-mono text-sm truncate">{service.providerAddress}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Transaction Status */}
                <div className={`bg-[#0d1117] border rounded-xl overflow-hidden transition-all ${
                  proofResult?.approved ? 'border-green-500 shadow-lg shadow-green-500/20' : 'border-gray-700'
                }`}>
                  <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-blue-500/10">
                    <h4 className="font-semibold flex items-center gap-2">
                      <DollarSign className="w-5 h-5 text-blue-400" />
                      x402 Payment Status
                    </h4>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      useRealPayment ? 'bg-red-600 text-white' : 'bg-gray-600 text-gray-200'
                    }`}>
                      {useRealPayment ? 'REAL USDC' : 'SIMULATED'}
                    </span>
                  </div>
                  <div className="p-4">
                    {proofResult?.approved ? (
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 p-4 bg-green-900/20 border border-green-800 rounded-lg">
                          <CheckCircle2 className="w-8 h-8 text-green-400" />
                          <div>
                            <div className="text-lg font-bold text-green-400">Payment Complete</div>
                            <div className="text-sm text-gray-400">
                              {useRealPayment ? 'Real USDC transferred on Arc testnet' : 'Simulated payment (no USDC spent)'}
                            </div>
                          </div>
                        </div>

                        {/* Real payment tx hash */}
                        {paymentResult && (
                          <a
                            href={paymentResult.explorerUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-between p-3 bg-green-500/10 border border-green-500/30 rounded-lg hover:bg-green-500/20 transition-colors"
                          >
                            <div>
                              <div className="text-xs text-gray-400 mb-1">Transaction Hash</div>
                              <div className="font-mono text-xs text-green-400 truncate">{paymentResult.txHash}</div>
                            </div>
                            <ExternalLink className="w-4 h-4 text-green-400 flex-shrink-0 ml-2" />
                          </a>
                        )}

                        <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                          <div className="text-xs text-gray-400 mb-1">Proof attested on-chain</div>
                          <div className="font-mono text-xs text-purple-400 truncate">{proofResult.proofHash}</div>
                        </div>

                        {/* Contract links */}
                        <div className="grid grid-cols-2 gap-2">
                          <a
                            href={`${ARC_CHAIN.explorerUrl}/address/${CONTRACTS.spendingGate}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 p-2 bg-gray-800/50 rounded-lg text-xs text-gray-400 hover:text-cyan-400 transition-colors"
                          >
                            <Shield className="w-3 h-3" />
                            SpendingGate
                            <ExternalLink className="w-2.5 h-2.5 ml-auto" />
                          </a>
                          <a
                            href={`${ARC_CHAIN.explorerUrl}/address/${CONTRACTS.usdc}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 p-2 bg-gray-800/50 rounded-lg text-xs text-gray-400 hover:text-green-400 transition-colors"
                          >
                            <DollarSign className="w-3 h-3" />
                            USDC Token
                            <ExternalLink className="w-2.5 h-2.5 ml-auto" />
                          </a>
                        </div>

                        <div className="text-center text-sm text-gray-400">
                          Robot is now charging...
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-8">
                        <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                        {paymentLoading ? 'Executing real USDC transfer...' : 'Processing payment...'}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Conclusion Phase */}
            {currentStep.phase === 'conclusion' && (
              <motion.div
                key="conclusion"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* Header */}
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex items-center gap-2 px-2 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                    <Bot className="w-3 h-3 text-cyan-400" />
                    <span className="text-cyan-400 font-medium text-xs">OpenMind</span>
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
                  Trustless Robot Commerce Complete
                </h2>
                <p className="text-gray-400 max-w-2xl mb-5 text-sm">
                  The robot purchased charging service using <span className="text-green-400">USDC</span> via <span className="text-blue-400">x402</span>.
                  <span className="text-yellow-400"> Jolt-Atlas</span> generated a SNARK proof verified by the payment gateway.
                  The proof was attested on <span className="text-purple-400">Arc</span> for audit trail.
                </p>

                {/* Stats */}
                <div className="grid grid-cols-4 gap-2 max-w-2xl mb-5">
                  <div className="p-2 bg-[#0d1117] border border-green-500/30 rounded-xl text-center">
                    <div className="text-lg font-bold text-green-400">${service.priceUsdc.toFixed(2)}</div>
                    <div className="text-[10px] text-gray-400">USDC Paid</div>
                  </div>
                  <div className="p-2 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                    <div className="text-lg font-bold text-yellow-400">~48KB</div>
                    <div className="text-[10px] text-gray-400">Proof Size</div>
                  </div>
                  <div className="p-2 bg-[#0d1117] border border-purple-500/30 rounded-xl text-center">
                    <div className="text-lg font-bold text-purple-400">Arc</div>
                    <div className="text-[10px] text-gray-400">Attested</div>
                  </div>
                  <div className="p-2 bg-[#0d1117] border border-cyan-500/30 rounded-xl text-center">
                    <div className="text-lg font-bold text-cyan-400">OM1</div>
                    <div className="text-[10px] text-gray-400">OpenMind</div>
                  </div>
                </div>

                {/* Workflow Recap */}
                <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    <span className="font-semibold text-white text-sm">What Happened</span>
                  </div>
                  <div className="space-y-3 text-xs">
                    <div className="flex items-start gap-3 p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
                      <Bot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="text-purple-400 font-medium">1. Robot Needed Service</span>
                        <p className="text-gray-400 mt-1">DeliveryBot-7 discovered low battery and found a ChargePoint station offering fast charging for $0.50 via x402.</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                      <Shield className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="text-green-400 font-medium">2. Policy Evaluation</span>
                        <p className="text-gray-400 mt-1">OpenMind LLM evaluated the request against owner-configured spending policy. Decision: {llmDecision?.decision.toUpperCase()} ({llmDecision ? `${(llmDecision.confidence * 100).toFixed(0)}%` : '...'})</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                      <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="text-yellow-400 font-medium">3. zkML Proof Generated</span>
                        <p className="text-gray-400 mt-1">Jolt-Atlas generated a ~48KB SNARK proof that the spending decision was computed correctly without revealing policy details.</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 bg-blue-900/20 rounded-lg border border-blue-500/30">
                      <DollarSign className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <span className="text-blue-400 font-medium">4. x402 Payment + Attestation</span>
                        <p className="text-gray-400 mt-1">Payment executed on Arc testnet. Proof hash attested on-chain for audit trail. Robot is now charging.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
