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
  Receipt,
  CheckCircle2,
  Shield,
  ExternalLink,
  Package,
  FileCheck,
  Wallet,
  Info,
  ArrowRight,
} from 'lucide-react';
import { useAccount } from 'wagmi';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useACKPayment } from '@/hooks/useACKPayment';
import { useCredentialSigning } from '@/hooks/useCredentialSigning';
import { ProofProgress } from './ProofProgress';
import {
  createDefaultInput,
  runSpendingModel,
  DEFAULT_SPENDING_POLICY,
  type SpendingModelInput,
} from '@/lib/spendingModel';
import { createAgentIdentity } from '@/lib/ack/identity';
import { createPaymentReceipt } from '@/lib/ack/payments';
import { formatDid, getIsoTimestamp, getExpirationTimestamp } from '@/lib/ack/client';
import { getExplorerTxUrl, ADDRESSES } from '@/lib/config';
import type { ACKAgentIdentity, ACKPaymentReceipt } from '@/lib/ack/types';
import type { SpendingProof } from '@/lib/types';

// Demo wallet address
const DEMO_WALLET_ADDRESS = '0x742d35Cc6634C0532925a3b844Bc9e7595f8fE00' as const;

// Phase definitions
const PHASES = ['intro', 'identity', 'proof', 'payment', 'receipt'] as const;
type Phase = typeof PHASES[number];

// Phase colors
const PHASE_COLORS: Record<Phase, string> = {
  intro: 'from-blue-500 to-cyan-500',
  identity: 'from-purple-500 to-pink-500',
  proof: 'from-yellow-500 to-orange-500',
  payment: 'from-green-500 to-emerald-500',
  receipt: 'from-cyan-500 to-blue-500',
};

const PHASE_LABELS: Record<Phase, string> = {
  intro: 'Introduction',
  identity: 'ACK-ID',
  proof: 'zkML Proof',
  payment: 'Payment',
  receipt: 'Receipt',
};

// Business annotation
interface BusinessAnnotation {
  title: string;
  takeaway: string;
  color: 'ack' | 'zkml' | 'arc' | 'combined';
  metric?: string;
  metricLabel?: string;
}

// Walkthrough step
interface WalkthroughStep {
  id: string;
  phase: Phase;
  title: string;
  description: string;
  ackNote?: string;
  docUrl?: string;
  docLabel?: string;
  duration: number;
  annotation?: BusinessAnnotation;
}

// Annotation colors
const ANNOTATION_COLORS = {
  ack: {
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/50',
    title: 'text-cyan-400',
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

// Walkthrough steps
const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'Catena ACK + zkML',
    description: 'Catena ACK provides the identity and receipt infrastructure for agentic commerce — verifiable agent identity (ACK-ID) and payment receipts (ACK-Pay) using W3C standards. zkML extends ACK with cryptographic proof that spending policies were correctly evaluated.',
    ackNote: 'ACK is open-source from Catena Labs - the infrastructure layer for AI agent commerce.',
    docUrl: 'https://catenalabs.com/projects/',
    docLabel: 'Catena Labs',
    duration: 5000,
    annotation: {
      title: 'ACK + zkML',
      takeaway: 'ACK verifies who the agent is. zkML verifies how the agent decided. Together: complete audit trail.',
      color: 'combined',
      metric: '3',
      metricLabel: 'verification layers',
    },
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'Complementary Capabilities',
    description: 'ACK provides agent identity (ACK-ID) and payment receipts (ACK-Pay). zkML adds cryptographic proof that the agent\'s spending policy — which checks vendor risk, budget limits, and compliance — was correctly evaluated before payment.',
    ackNote: 'ACK uses W3C DIDs and Verifiable Credentials. zkML uses SNARK proofs for spending policy verification.',
    duration: 5000,
    annotation: {
      title: 'What Each Provides',
      takeaway: 'ACK: "Who is this agent?" zkML: "Did it check vendor risk, budget, and compliance?"',
      color: 'ack',
      metric: 'W3C',
      metricLabel: 'standards',
    },
  },
  {
    id: 'identity-1',
    phase: 'identity',
    title: 'Creating Agent Identity',
    description: 'The agent creates a W3C Decentralized Identifier (DID) linked to its owner. This establishes a verifiable identity that can be cryptographically proven.',
    ackNote: 'ACK-ID uses did:key method for simplicity. Production can use did:web or did:ion.',
    docUrl: 'https://github.com/agentcommercekit/ack',
    docLabel: 'ACK-ID Spec',
    duration: 6000,
    annotation: {
      title: 'Decentralized Identity',
      takeaway: 'No central authority needed. Agent identity is self-sovereign and verifiable.',
      color: 'ack',
      metric: 'DID',
      metricLabel: 'W3C standard',
    },
  },
  {
    id: 'identity-2',
    phase: 'identity',
    title: 'Controller Credential Issued',
    description: 'A ControllerCredential links the agent DID to its human/org owner. This proves who authorized the agent and is liable for its actions.',
    ackNote: 'Verifiable Credential proves ownership without exposing private keys.',
    duration: 5000,
  },
  {
    id: 'proof-1',
    phase: 'proof',
    title: 'Spending Policy Evaluation',
    description: 'The ML spending policy checks if a purchase should be approved by evaluating vendor risk, budget constraints, track record, and compliance status together.',
    ackNote: 'This exact model execution will be proven - same inputs, same computation, same output.',
    duration: 6000,
    annotation: {
      title: 'Deterministic Spending Policy',
      takeaway: 'The spending policy model is deterministic: given inputs X, it ALWAYS produces output Y. This is what we prove.',
      color: 'zkml',
      metric: '6',
      metricLabel: 'spending factors',
    },
  },
  {
    id: 'proof-2',
    phase: 'proof',
    title: 'Generating SNARK Proof',
    description: 'JOLT-Atlas runs the spending policy ONNX model inside a SNARK circuit, generating a ~48KB cryptographic proof that the exact model ran on the exact inputs.',
    ackNote: 'Proof generation takes 4-12 seconds. The proof is unforgeable.',
    duration: 8000,
  },
  {
    id: 'proof-3',
    phase: 'proof',
    title: 'Local Proof Verification',
    description: 'The proof is verified locally (off-chain) in <150ms. This verification result will be attested on-chain to gate the SpendingGateWallet.',
    ackNote: 'Verification is instant. The verificationHash (not just proofHash) will be attested on-chain.',
    duration: 5000,
    annotation: {
      title: 'Verification Ready for Attestation',
      takeaway: 'Proof verified OFF-CHAIN. Next: attest verificationHash ON-CHAIN to unlock SpendingGateWallet.',
      color: 'zkml',
      metric: '<150ms',
      metricLabel: 'verification',
    },
  },
  {
    id: 'payment-1',
    phase: 'payment',
    title: 'On-Chain Verification Attestation',
    description: 'Submitting verificationHash to ProofAttestation contract. This attests that verification PASSED (not just that a proof exists). verificationHash = hash(proofHash + decision + confidence + timestamp).',
    ackNote: 'Live Mode: Real attestation tx. We attest the verification result, not just the proof.',
    docUrl: 'https://arc.network',
    docLabel: 'Arc Network',
    duration: 6000,
    annotation: {
      title: 'Verification Attestation',
      takeaway: 'Attesting that verification PASSED. SpendingGateWallet checks this before releasing funds.',
      color: 'arc',
      metric: 'Required',
      metricLabel: 'for gated transfer',
    },
  },
  {
    id: 'payment-2',
    phase: 'payment',
    title: 'Gated Transfer Executed',
    description: 'SpendingGateWallet verified the attested verificationHash on-chain, then released $0.01 USDC. Full trustless flow: verification result triggers payment.',
    ackNote: 'Live Mode: 2 real transactions — verification attestation, then gated transfer.',
    duration: 6000,
    annotation: {
      title: 'SpendingGateWallet',
      takeaway: 'Smart contract enforces: no attested verification = no funds released. Trustless.',
      color: 'arc',
      metric: '$0.01',
      metricLabel: 'gated USDC',
    },
  },
  {
    id: 'receipt-1',
    phase: 'receipt',
    title: 'Issuing Payment Receipt',
    description: 'ACK-Pay issues a PaymentReceiptCredential - a W3C Verifiable Credential proving the payment was made by a verified agent following approved policy.',
    ackNote: 'Receipt links: Agent DID + Proof Hash + Tx Hash. Complete audit chain.',
    docUrl: 'https://github.com/agentcommercekit/ack',
    docLabel: 'ACK-Pay Spec',
    duration: 6000,
    annotation: {
      title: 'Audit-Ready Receipt',
      takeaway: 'Verifiable credential proves: who paid, what policy, which transaction. Forever.',
      color: 'combined',
      metric: 'VC',
      metricLabel: 'standard',
    },
  },
  {
    id: 'receipt-2',
    phase: 'receipt',
    title: 'Complete: Full Audit Trail',
    description: 'The agent now has a complete, cryptographically verifiable audit trail: ACK-ID identity, zkML policy proof, Arc transaction, and ACK-Pay receipt.',
    duration: 5000,
  },
];

type DemoMode = 'demo' | 'live';

export function ACKWalkthrough() {
  // Mode
  const [mode, setMode] = useState<DemoMode>('demo');
  const { address, isConnected } = useAccount();
  const walletAddress = mode === 'live' && isConnected && address ? address : DEMO_WALLET_ADDRESS;

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);
  const processedStepRef = useRef<string | null>(null);

  // Demo state
  const [identity, setIdentity] = useState<ACKAgentIdentity | null>(null);
  const [spendingProof, setSpendingProof] = useState<SpendingProof | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<ACKPaymentReceipt | null>(null);
  const [agentThoughts, setAgentThoughts] = useState<string[]>([]);

  // Annotation overlay
  const [showingAnnotation, setShowingAnnotation] = useState(false);
  const [annotationData, setAnnotationData] = useState<{ annotation: BusinessAnnotation; stepTitle: string } | null>(null);
  const annotationTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Proof generation
  const { state: proofState, generateProof, reset: resetProof } = useProofGeneration();
  const [input] = useState<SpendingModelInput>(createDefaultInput());

  // Real payment (Live Mode)
  const {
    executeTransfer,
    executeGatedTransfer,
    isPending: isPaymentPending,
    isConfirming: isPaymentConfirming,
    isConfirmed: isPaymentConfirmed,
    txHash: liveTxHash,
    gatedResult,
    isGatedPending,
    formattedBalance,
    reset: resetPayment,
  } = useACKPayment(mode === 'live' && isConnected && address ? address : undefined);

  // Track attestation details from gated transfer
  const [attestationTxHash, setAttestationTxHash] = useState<string | null>(null);
  const [verificationHash, setVerificationHash] = useState<string | null>(null);

  // Credential signing (Live Mode)
  const {
    signControllerCredential,
    signPaymentReceipt,
    signature: credentialSignature,
    isPending: isSigningPending,
    isSuccess: isSigningSuccess,
    reset: resetSigning,
  } = useCredentialSigning();

  // Track which credential we're signing
  const [signingType, setSigningType] = useState<'identity' | 'receipt' | null>(null);
  const [pendingIdentity, setPendingIdentity] = useState<ACKAgentIdentity | null>(null);
  const [pendingReceipt, setPendingReceipt] = useState<ACKPaymentReceipt | null>(null);

  const currentStep = WALKTHROUGH_STEPS[currentStepIndex];
  const currentPhaseIndex = PHASES.indexOf(currentStep.phase);

  // Keep ref in sync
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Update txHash when live payment confirms (direct transfer mode - fallback)
  useEffect(() => {
    if (mode === 'live' && liveTxHash && isPaymentConfirmed && !txHash) {
      setTxHash(liveTxHash);
    }
  }, [mode, liveTxHash, isPaymentConfirmed, txHash]);

  // Update state when gated transfer completes
  useEffect(() => {
    if (gatedResult?.success && gatedResult.transfer) {
      setTxHash(gatedResult.transfer.txHash);
      setAttestationTxHash(gatedResult.transfer.attestationTxHash || null);
      setVerificationHash(gatedResult.transfer.verificationHash || null);
    }
  }, [gatedResult]);

  // Handle credential signature completion
  useEffect(() => {
    if (isSigningSuccess && credentialSignature) {
      if (signingType === 'identity' && pendingIdentity) {
        // Update identity with real signature
        const signedIdentity: ACKAgentIdentity = {
          ...pendingIdentity,
          controllerCredential: {
            ...pendingIdentity.controllerCredential,
            proof: {
              ...pendingIdentity.controllerCredential.proof!,
              proofValue: credentialSignature,
            },
          },
        };
        setIdentity(signedIdentity);
        setPendingIdentity(null);
        setAgentThoughts(prev => [...prev, `Signature: ${credentialSignature.slice(0, 14)}...`]);
        setAgentThoughts(prev => [...prev, 'Identity verified with wallet signature.']);
      } else if (signingType === 'receipt' && pendingReceipt) {
        // Update receipt with real signature
        const signedReceipt: ACKPaymentReceipt = {
          ...pendingReceipt,
          receiptCredential: {
            ...pendingReceipt.receiptCredential,
            proof: {
              ...pendingReceipt.receiptCredential.proof!,
              proofValue: credentialSignature,
            },
          },
        };
        setReceipt(signedReceipt);
        setPendingReceipt(null);
      }
      setSigningType(null);
      resetSigning();
    }
  }, [isSigningSuccess, credentialSignature, signingType, pendingIdentity, pendingReceipt, resetSigning]);

  // Main playback effect
  useEffect(() => {
    if (!isPlaying) return;

    const step = WALKTHROUGH_STEPS[currentStepIndex];
    const stepKey = `${step.id}-${isPlaying}`;
    const alreadyProcessed = processedStepRef.current === stepKey;

    if (!alreadyProcessed) {
      processedStepRef.current = stepKey;

      // Phase-specific effects
      if (step.phase === 'identity' && step.id === 'identity-1') {
        // Create identity
        setAgentThoughts(['Generating DID from wallet address...']);
        setTimeout(() => {
          if (isPlayingRef.current) {
            const newIdentity = createAgentIdentity(walletAddress, 'Spending Agent');

            if (mode === 'live' && isConnected && address) {
              // Live Mode: Request wallet signature for credential
              setPendingIdentity(newIdentity);
              setSigningType('identity');
              setAgentThoughts(prev => [...prev, `DID: ${formatDid(newIdentity.did)}`]);
              setAgentThoughts(prev => [...prev, 'Requesting wallet signature...']);

              const issuanceDate = getIsoTimestamp();
              const expirationDate = getExpirationTimestamp();

              signControllerCredential({
                agentDid: newIdentity.did,
                agentName: newIdentity.name,
                controller: address,
                issuanceDate,
                expirationDate,
              });
            } else {
              // Demo Mode: Use simulated credential
              setIdentity(newIdentity);
              setAgentThoughts(prev => [...prev, `DID: ${formatDid(newIdentity.did)}`]);
            }
          }
        }, 1500);
        setTimeout(() => {
          if (isPlayingRef.current && mode === 'demo') {
            setAgentThoughts(prev => [...prev, 'Issuing ControllerCredential...']);
          }
        }, 2500);
        setTimeout(() => {
          if (isPlayingRef.current && mode === 'demo') {
            setAgentThoughts(prev => [...prev, 'Identity verified and ready.']);
          }
        }, 3500);
      }

      if (step.phase === 'proof' && step.id === 'proof-1') {
        // Generate proof
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
          }
        });
      }

      if (step.phase === 'payment' && step.id === 'payment-1' && spendingProof) {
        if (mode === 'live' && isConnected && address) {
          // Live Mode: Execute gated transfer via SpendingGateWallet
          // 1. Attests verificationHash (not just proofHash)
          // 2. Executes gatedTransfer (checks attestation on-chain)
          executeGatedTransfer(
            ADDRESSES.demoMerchant,
            '0.01',
            spendingProof.proofHash,
            {
              decision: spendingProof.decision.shouldBuy,
              confidence: spendingProof.decision.confidence,
            },
            identity?.did
          );
        } else {
          // Demo Mode: Simulate attestation + gated transfer
          setTimeout(() => {
            if (isPlayingRef.current) {
              // Mock attestation tx
              const mockAttestationTxHash = `0x${Array.from({ length: 64 }, () =>
                Math.floor(Math.random() * 16).toString(16)
              ).join('')}`;
              setAttestationTxHash(mockAttestationTxHash);
              // Mock verification hash
              const mockVerificationHash = `0x${Array.from({ length: 64 }, () =>
                Math.floor(Math.random() * 16).toString(16)
              ).join('')}`;
              setVerificationHash(mockVerificationHash);
            }
          }, 2000);
          setTimeout(() => {
            if (isPlayingRef.current) {
              // Mock transfer tx (after attestation)
              const mockTxHash = `0x${Array.from({ length: 64 }, () =>
                Math.floor(Math.random() * 16).toString(16)
              ).join('')}`;
              setTxHash(mockTxHash);
            }
          }, 4000);
        }
      }

      if (step.phase === 'receipt' && step.id === 'receipt-1' && identity && spendingProof && txHash) {
        // Issue receipt
        setTimeout(() => {
          if (isPlayingRef.current) {
            const paymentReceipt = createPaymentReceipt({
              txHash,
              amount: '0.01',
              recipient: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',
              proofHash: spendingProof.proofHash,
              agentDid: identity.did,
              ownerAddress: walletAddress,
            });

            if (mode === 'live' && isConnected && address) {
              // Live Mode: Request wallet signature for receipt
              setPendingReceipt(paymentReceipt);
              setSigningType('receipt');

              const issuanceDate = getIsoTimestamp();

              signPaymentReceipt({
                agentDid: identity.did,
                txHash: txHash as `0x${string}`,
                amount: '0.01',
                recipient: ADDRESSES.demoMerchant,
                proofHash: spendingProof.proofHash as `0x${string}`,
                issuanceDate,
              });
            } else {
              // Demo Mode: Use simulated receipt
              setReceipt(paymentReceipt);
            }
          }
        }, 2000);
      }
    }

    // Timer for advancing - show annotation after a delay so user can read main content first
    timerRef.current = setTimeout(() => {
      if (isPlayingRef.current) {
        if (step.annotation) {
          // Delay showing annotation by 2 seconds after step content is shown
          setTimeout(() => {
            if (isPlayingRef.current) {
              setAnnotationData({ annotation: step.annotation, stepTitle: step.title });
              setShowingAnnotation(true);

              annotationTimerRef.current = setTimeout(() => {
                setShowingAnnotation(false);
                setAnnotationData(null);
                advanceStep();
              }, 4000);
            }
          }, 2000);
        } else {
          advanceStep();
        }
      }
    }, step.duration);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
    };
  }, [currentStepIndex, isPlaying, generateProof, input, identity, spendingProof, txHash, walletAddress, mode, isConnected, address, executeTransfer, signControllerCredential, signPaymentReceipt]);

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
    setIdentity(null);
    setSpendingProof(null);
    setTxHash(null);
    setAttestationTxHash(null);
    setVerificationHash(null);
    setReceipt(null);
    setAgentThoughts([]);
    setShowingAnnotation(false);
    setAnnotationData(null);
    setSigningType(null);
    setPendingIdentity(null);
    setPendingReceipt(null);
    processedStepRef.current = null;
    resetProof();
    resetPayment();
    resetSigning();
    if (timerRef.current) clearTimeout(timerRef.current);
    if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
  }, [resetProof, resetPayment, resetSigning]);

  const handleModeChange = (newMode: DemoMode) => {
    if (newMode !== mode) {
      handleReset();
      setMode(newMode);
    }
  };

  const getPhaseIcon = (phase: Phase) => {
    switch (phase) {
      case 'intro': return <Info className="w-4 h-4" />;
      case 'identity': return <User className="w-4 h-4" />;
      case 'proof': return <Zap className="w-4 h-4" />;
      case 'payment': return <CreditCard className="w-4 h-4" />;
      case 'receipt': return <Receipt className="w-4 h-4" />;
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
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30'
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
            {/* Mode Toggle */}
            <div className="flex bg-gray-800 rounded-lg p-0.5 border border-gray-700">
              <button
                onClick={() => handleModeChange('demo')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  mode === 'demo' ? 'bg-purple-500/30 text-purple-400' : 'text-gray-400 hover:text-white'
                }`}
              >
                Demo
              </button>
              <button
                onClick={() => handleModeChange('live')}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  mode === 'live' ? 'bg-green-500/30 text-green-400' : 'text-gray-400 hover:text-white'
                }`}
              >
                Live
              </button>
            </div>
            {mode === 'live' && <ConnectButton />}

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
              Verifiable Agent Commerce via<br />
              <span className="text-cyan-400 font-medium">Catena ACK + zkML + Arc</span>
            </p>
          </div>

          {/* Current Step */}
          <div className="p-4 flex-1 overflow-y-auto">
            {/* Phase Badge */}
            <div className="mb-2">
              <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${PHASE_COLORS[currentStep.phase]} text-white`}>
                {PHASE_LABELS[currentStep.phase]}
              </span>
            </div>

            {/* Title & Description */}
            <h3 className="text-lg font-bold text-white mb-2">{currentStep.title}</h3>
            <p className="text-sm text-gray-400 leading-relaxed mb-3">{currentStep.description}</p>

            {/* ACK Note */}
            {currentStep.ackNote && (
              <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg mb-3">
                <div className="flex items-start gap-2">
                  <Package className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-xs text-gray-300">{currentStep.ackNote}</p>
                    {currentStep.docUrl && (
                      <a
                        href={currentStep.docUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-1.5 text-[10px] text-cyan-400 hover:underline"
                      >
                        <span>{currentStep.docLabel || 'Documentation'}</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
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
              <a href="https://catenalabs.com/projects/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-cyan-400 transition-colors">
                Catena Labs ACK <ExternalLink className="w-2.5 h-2.5" />
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
                <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                  <Package className="w-4 h-4 text-cyan-400" />
                  <span className="text-cyan-400 font-medium text-sm">Catena ACK</span>
                </div>
                <span className="text-gray-600">+</span>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="text-yellow-400 font-medium text-sm">JOLT-Atlas zkML</span>
                </div>
                <span className="text-gray-600">+</span>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <Shield className="w-4 h-4 text-purple-400" />
                  <span className="text-purple-400 font-medium text-sm">Arc Testnet</span>
                </div>
              </div>

              <h2 className="text-2xl font-bold mb-2 text-white">Verifiable Agent Commerce</h2>
              <p className="text-gray-400 max-w-2xl mb-6 text-sm">
                <a href="https://catenalabs.com/projects/" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">Catena ACK</a> provides
                the identity and receipt infrastructure for agentic commerce. zkML extends ACK with cryptographic proof that spending policies were correctly evaluated.
              </p>

              {/* Three Pillars */}
              <div className="grid grid-cols-3 gap-4 max-w-3xl">
                <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <User className="w-5 h-5 text-cyan-400" />
                      <span className="text-cyan-400 font-semibold">ACK-ID</span>
                    </div>
                    <span className="text-[9px] text-cyan-400/60 bg-cyan-500/10 px-1.5 py-0.5 rounded">Catena</span>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">Verifiable agent identity using W3C DIDs</p>
                  <div className="space-y-1.5 text-[10px]">
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      <span>Decentralized Identifiers</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      <span>Controller Credentials</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      <span>Owner Verification</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Zap className="w-5 h-5 text-yellow-400" />
                      <span className="text-yellow-400 font-semibold">zkML Proof</span>
                    </div>
                    <span className="text-[9px] text-yellow-400/60 bg-yellow-500/10 px-1.5 py-0.5 rounded">Extends</span>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">The ML spending policy checks vendor risk, budget limits, and compliance before approving purchases</p>
                  <div className="space-y-1.5 text-[10px]">
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>6-Factor Decision Model</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>SNARK Proof (~48KB)</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-yellow-400" />
                      <span>Cryptographic Verification</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Receipt className="w-5 h-5 text-green-400" />
                      <span className="text-green-400 font-semibold">ACK-Pay</span>
                    </div>
                    <span className="text-[9px] text-green-400/60 bg-green-500/10 px-1.5 py-0.5 rounded">Catena</span>
                  </div>
                  <p className="text-xs text-gray-400 mb-3">Verifiable payment receipts for audit</p>
                  <div className="space-y-1.5 text-[10px]">
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-green-400" />
                      <span>Verifiable Credentials</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-green-400" />
                      <span>Proof + Tx Linkage</span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-500">
                      <CheckCircle2 className="w-3 h-3 text-green-400" />
                      <span>Permanent Audit Trail</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Identity Phase */}
          {currentStep.phase === 'identity' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">Creating Agent Identity</h2>

              {/* Agent Thoughts */}
              <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4 mb-6 max-w-xl">
                <div className="flex items-center gap-2 mb-3">
                  <User className="w-5 h-5 text-purple-400" />
                  <span className="text-purple-400 font-semibold text-sm">Identity Generation</span>
                </div>
                <div className="space-y-2 font-mono text-xs">
                  {agentThoughts.map((thought, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <ArrowRight className="w-3 h-3 text-purple-400 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-300">{thought}</span>
                    </div>
                  ))}
                  {isPlaying && agentThoughts.length < 4 && (
                    <div className="flex items-center gap-2 text-gray-500">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                      <span>Processing...</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Signing Status (Live Mode) */}
              {mode === 'live' && isSigningPending && signingType === 'identity' && (
                <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4 max-w-xl mb-4">
                  <div className="flex items-center gap-2 text-yellow-400 text-sm">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                    <span>Please sign the credential in your wallet...</span>
                  </div>
                </div>
              )}

              {/* Identity Card */}
              {identity && (
                <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-4 max-w-xl">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 font-semibold text-sm">Identity Created</span>
                    {mode === 'live' && identity.controllerCredential.proof?.proofValue.length === 132 && (
                      <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full ml-2">
                        Wallet Signed
                      </span>
                    )}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Agent Name</span>
                      <span className="text-white font-medium">{identity.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">DID</span>
                      <span className="text-cyan-400 font-mono text-xs">{formatDid(identity.did)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Owner</span>
                      <span className="text-white font-mono text-xs">
                        {identity.ownerAddress.slice(0, 6)}...{identity.ownerAddress.slice(-4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Credential</span>
                      <span className="text-purple-400 text-xs">ControllerCredential</span>
                    </div>
                    {mode === 'live' && identity.controllerCredential.proof?.proofValue.length === 132 && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Signature</span>
                        <span className="text-green-400 font-mono text-xs">
                          {identity.controllerCredential.proof.proofValue.slice(0, 14)}...
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Proof Phase */}
          {currentStep.phase === 'proof' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                {currentStep.id === 'proof-1' ? 'Spending Policy Inputs' :
                 currentStep.id === 'proof-2' ? 'Generating SNARK Proof' :
                 'Local Proof Verification'}
              </h2>

              {/* Spending Policy Inputs (shown first) */}
              {currentStep.id === 'proof-1' && !spendingProof && (
                <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4 max-w-xl mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-5 h-5 text-purple-400" />
                    <span className="text-purple-400 font-semibold text-sm">ML Spending Policy</span>
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
                    zkML proves this exact policy model ran on these exact inputs — unforgeable verification.
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
                      <span className="text-green-400 font-semibold text-sm">Proof Generated</span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Spending Policy Decision</span>
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

                  {/* Local Verification Status */}
                  {currentStep.id === 'proof-3' && (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Spending Policy Verification Passed</span>
                      </div>
                      <p className="text-xs text-gray-400 mb-2">
                        Spending policy proof verified off-chain in &lt;150ms. Ready for on-chain attestation.
                      </p>
                      <div className="p-2 bg-purple-500/10 border border-purple-500/20 rounded text-xs text-purple-400">
                        Next: Attest verification on-chain → SpendingGateWallet releases funds.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Payment Phase */}
          {currentStep.phase === 'payment' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">
                {currentStep.id === 'payment-1' ? 'Verification Attestation' : 'Gated Transfer'}
              </h2>

              {/* Live Mode: Show balance */}
              {mode === 'live' && isConnected && (
                <div className="mb-4 flex items-center gap-2 text-sm">
                  <Wallet className="w-4 h-4 text-gray-400" />
                  <span className="text-gray-400">USDC Balance:</span>
                  <span className="text-white font-mono">${formattedBalance}</span>
                </div>
              )}

              {/* Step 1: Verification Attestation */}
              {currentStep.id === 'payment-1' && (
                <div className="space-y-4 max-w-xl">
                  {/* What's being attested */}
                  <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <FileCheck className="w-5 h-5 text-purple-400" />
                      <span className="text-purple-400 font-semibold text-sm">Attesting Verification Result</span>
                    </div>
                    <p className="text-xs text-gray-400 mb-3">
                      We attest the <span className="text-yellow-400">verificationHash</span>, not just the proof.
                      This captures that verification passed, not just that a proof exists.
                    </p>
                    <div className="bg-gray-900/50 p-2 rounded text-xs font-mono text-gray-300">
                      verificationHash = keccak256(proofHash, decision, confidence, timestamp)
                    </div>
                  </div>

                  {/* Attestation Status */}
                  {!attestationTxHash ? (
                    <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 text-yellow-400 text-sm">
                        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                        <span>
                          {isGatedPending
                            ? 'Submitting verification attestation to Arc...'
                            : 'Preparing verification attestation...'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Verification Attested</span>
                        {mode === 'live' && (
                          <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full ml-2">
                            On-Chain
                          </span>
                        )}
                      </div>
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
                    </div>
                  )}
                </div>
              )}

              {/* Step 2: Gated Transfer */}
              {currentStep.id === 'payment-2' && (
                <div className="space-y-4 max-w-xl">
                  {/* Attestation summary */}
                  {attestationTxHash && (
                    <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3">
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <CheckCircle2 className="w-4 h-4 text-green-400" />
                        <span>Verification attested on-chain</span>
                        <span className="text-gray-600">•</span>
                        <span className="text-yellow-400 font-mono">{verificationHash?.slice(0, 10)}...</span>
                      </div>
                    </div>
                  )}

                  {/* Transfer Status */}
                  {!txHash ? (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <CreditCard className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Executing Gated Transfer</span>
                      </div>
                      <div className="flex items-center gap-2 text-gray-400 text-sm">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                        <span>SpendingGateWallet checking attestation...</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Smart contract verifies the attestation exists before releasing funds.
                      </p>
                    </div>
                  ) : (
                    <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400 font-semibold text-sm">Gated Transfer Complete</span>
                        {mode === 'live' && (
                          <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full ml-2">
                            Real USDC
                          </span>
                        )}
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
                        <div className="flex justify-between">
                          <span className="text-gray-400">Network</span>
                          <span className="text-purple-400">Arc Testnet</span>
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
                        SpendingGateWallet verified the attestation on-chain before releasing funds.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Receipt Phase */}
          {currentStep.phase === 'receipt' && (
            <div>
              <h2 className="text-2xl font-bold mb-4 text-white">Payment Receipt</h2>

              {/* Signing Status (Live Mode) */}
              {mode === 'live' && isSigningPending && signingType === 'receipt' && (
                <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4 max-w-xl mb-4">
                  <div className="flex items-center gap-2 text-yellow-400 text-sm">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                    <span>Please sign the receipt in your wallet...</span>
                  </div>
                </div>
              )}

              {!receipt && !isSigningPending && (
                <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-4 max-w-xl">
                  <div className="flex items-center gap-2 mb-3">
                    <Receipt className="w-5 h-5 text-cyan-400" />
                    <span className="text-cyan-400 font-semibold text-sm">Issuing Receipt</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-400 text-sm">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                    <span>Creating PaymentReceiptCredential...</span>
                  </div>
                </div>
              )}

              {receipt && (
                <div className="space-y-4 max-w-xl">
                  <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <FileCheck className="w-5 h-5 text-green-400" />
                      <span className="text-green-400 font-semibold text-sm">Receipt Issued</span>
                      {mode === 'live' && receipt.receiptCredential.proof?.proofValue.length === 132 && (
                        <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full ml-2">
                          Wallet Signed
                        </span>
                      )}
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Credential Type</span>
                        <span className="text-purple-400">PaymentReceiptCredential</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Amount</span>
                        <span className="text-white">${receipt.amount} USDC</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Transaction</span>
                        <span className="text-cyan-400 font-mono text-xs">
                          {receipt.txHash.slice(0, 10)}...
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Proof Hash</span>
                        <span className="text-yellow-400 font-mono text-xs">
                          {receipt.proofHash.slice(0, 10)}...
                        </span>
                      </div>
                      {mode === 'live' && receipt.receiptCredential.proof?.proofValue.length === 132 && (
                        <div className="flex justify-between">
                          <span className="text-gray-400">Signature</span>
                          <span className="text-green-400 font-mono text-xs">
                            {receipt.receiptCredential.proof.proofValue.slice(0, 14)}...
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Complete Summary - Expanded like Crossmint */}
                  {currentStep.id === 'receipt-2' && (
                    <div className="space-y-4">
                      {/* Partner Badges */}
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 px-2 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                          <Package className="w-3 h-3 text-cyan-400" />
                          <span className="text-cyan-400 font-medium text-xs">Catena ACK</span>
                        </div>
                        <span className="text-gray-600 text-xs">+</span>
                        <div className="flex items-center gap-2 px-2 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                          <Zap className="w-3 h-3 text-yellow-400" />
                          <span className="text-yellow-400 font-medium text-xs">Jolt-Atlas</span>
                        </div>
                        <span className="text-gray-600 text-xs">+</span>
                        <div className="flex items-center gap-2 px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                          <Shield className="w-3 h-3 text-purple-400" />
                          <span className="text-purple-400 font-medium text-xs">Arc Testnet</span>
                        </div>
                      </div>

                      {/* Title */}
                      <div>
                        <h3 className="text-lg font-bold text-white mb-1">Verifiable Agent Commerce Complete</h3>
                        <p className="text-gray-400 text-sm">
                          Agent executed <span className="text-green-400">$0.01 USDC</span> transfer with cryptographic proof of spending policy compliance.
                          Verification attested on-chain, <span className="text-purple-400">SpendingGateWallet</span> verified before releasing funds.
                        </p>
                      </div>

                      {/* Stats Grid */}
                      <div className="grid grid-cols-4 gap-2">
                        <div className="p-2 bg-[#0d1117] border border-green-500/30 rounded-xl text-center">
                          <div className="text-lg font-bold text-green-400">$0.01</div>
                          <div className="text-[10px] text-gray-400">Transfer Value</div>
                        </div>
                        <div className="p-2 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                          <div className="text-lg font-bold text-yellow-400">6</div>
                          <div className="text-[10px] text-gray-400">Spending Factors</div>
                        </div>
                        <div className="p-2 bg-[#0d1117] border border-cyan-500/30 rounded-xl text-center">
                          <div className="text-lg font-bold text-cyan-400">~48KB</div>
                          <div className="text-[10px] text-gray-400">SNARK Proof</div>
                        </div>
                        <div className="p-2 bg-[#0d1117] border border-purple-500/30 rounded-xl text-center">
                          <div className="text-lg font-bold text-purple-400">2</div>
                          <div className="text-[10px] text-gray-400">On-Chain Txs</div>
                        </div>
                      </div>

                      {/* What Happened */}
                      <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <CheckCircle2 className="w-4 h-4 text-green-400" />
                          <span className="font-semibold text-white text-sm">What Happened</span>
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex items-start gap-3 p-2 bg-cyan-900/20 rounded-lg border border-cyan-500/30">
                            <User className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                            <div>
                              <span className="text-cyan-400 font-medium">1. ACK-ID: Verifiable agent identity established</span>
                              <p className="text-gray-400 mt-0.5">W3C DID + ControllerCredential signed by owner wallet.</p>
                            </div>
                          </div>
                          <div className="flex items-start gap-3 p-2 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                            <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                            <div>
                              <span className="text-yellow-400 font-medium">2. zkML: Spending policy cryptographically verified</span>
                              <p className="text-gray-400 mt-0.5">SNARK proves policy checked vendor risk, budget, and compliance correctly.</p>
                            </div>
                          </div>
                          <div className="flex items-start gap-3 p-2 bg-purple-900/20 rounded-lg border border-purple-500/30">
                            <Shield className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                            <div>
                              <span className="text-purple-400 font-medium">3. On-chain: Verification result attested to Arc</span>
                              <p className="text-gray-400 mt-0.5">verificationHash submitted to ProofAttestation contract.</p>
                            </div>
                          </div>
                          <div className="flex items-start gap-3 p-2 bg-green-900/20 rounded-lg border border-green-500/30">
                            <CreditCard className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                            <div>
                              <span className="text-green-400 font-medium">4. ACK-Pay: Gated transfer executed, receipt issued</span>
                              <p className="text-gray-400 mt-0.5">SpendingGateWallet verified attestation, then released USDC.</p>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* What This Demonstrates */}
                      <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-cyan-500/30 rounded-xl p-4">
                        <div className="text-sm font-medium text-cyan-400 mb-2">What This Demonstrates</div>
                        <p className="text-gray-300 text-xs">
                          An AI agent executed a transfer with <span className="text-cyan-400">verifiable identity</span> (ACK-ID),
                          <span className="text-yellow-400"> cryptographic proof of spending policy compliance</span> (zkML),
                          <span className="text-purple-400"> on-chain attestation</span>, and an
                          <span className="text-green-400"> auditable receipt</span> (ACK-Pay).
                        </p>
                      </div>

                      {/* Links */}
                      <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3">
                        <div className="flex flex-wrap items-center gap-2 text-[10px]">
                          <span className="text-gray-500">Docs:</span>
                          <a
                            href="https://catenalabs.com/projects/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 bg-cyan-500/10 text-cyan-400 rounded border border-cyan-500/30 hover:border-cyan-400 transition-colors flex items-center gap-1"
                          >
                            Catena ACK <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                          <a
                            href="https://github.com/ICME-Lab/jolt-atlas"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 bg-yellow-500/10 text-yellow-400 rounded border border-yellow-500/30 hover:border-yellow-400 transition-colors flex items-center gap-1"
                          >
                            JOLT-Atlas <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                          <span className="text-gray-700">|</span>
                          <span className="text-gray-500">Contracts:</span>
                          <a
                            href="https://testnet.arcscan.app/address/0xBE9a5DF7C551324CB872584C6E5bF56799787952"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 bg-purple-500/10 text-purple-400 rounded border border-purple-500/30 hover:border-purple-400 transition-colors flex items-center gap-1"
                          >
                            ProofAttestation <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                          <a
                            href="https://testnet.arcscan.app/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 bg-green-500/10 text-green-400 rounded border border-green-500/30 hover:border-green-400 transition-colors flex items-center gap-1"
                          >
                            SpendingGateWallet <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
