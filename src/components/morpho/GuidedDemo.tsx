'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play, Pause, SkipForward, RotateCcw,
  ChevronRight, Zap, Shield, Bot, CheckCircle, ExternalLink, Globe, Radio, Cpu, ToggleLeft, ToggleRight, Wifi, WifiOff
} from 'lucide-react';
import { SpendingPolicy, AgentConfig, Transaction, ProofState, MorphoOperation } from '@/lib/morpho/types';
import { MOCK_MARKETS, AGENT_DECISIONS, generateMockTxHash, generateMockProofHash, DEFAULT_POLICY, NETWORK_INFO } from '@/lib/morpho/mockData';
import { generateZkmlProof, checkProverHealth, getProverInfo } from '@/lib/morpho/zkmlProver';

interface WorkflowStep {
  id: string;
  phase: 'intro' | 'policy' | 'authorize' | 'agent' | 'complete';
  title: string;
  description: string;
  highlight: 'morpho' | 'zkml' | 'policy' | 'proof';
  duration: number;
}

const OPERATION_NAMES = ['SUPPLY', 'BORROW', 'WITHDRAW', 'REPAY'];

const HIGHLIGHT_COLORS = {
  morpho: 'from-blue-500 to-cyan-500',
  zkml: 'from-purple-500 to-pink-500',
  policy: 'from-green-500 to-emerald-500',
  proof: 'from-orange-500 to-yellow-500',
};

const HIGHLIGHT_LABELS = {
  morpho: 'Morpho Integration',
  zkml: 'zkML Technology',
  policy: 'Policy Enforcement',
  proof: 'Proof Verification',
};

const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'Welcome to Morpho zkML Spending Proofs',
    description: 'This workflow shows how AI agents can autonomously manage Morpho vault positions while cryptographically proving every action complies with owner-defined policies.',
    highlight: 'morpho',
    duration: 5000,
  },
  {
    id: 'intro-2',
    phase: 'intro',
    title: 'The Problem: Trustless Autonomous DeFi',
    description: 'How can vault owners trust AI agents to manage their funds? Traditional access controls are binary - either full access or none. We need cryptographic guarantees.',
    highlight: 'zkml',
    duration: 5000,
  },
  {
    id: 'policy-1',
    phase: 'policy',
    title: 'Step 1: Define Spending Policy',
    description: 'The vault owner defines constraints: daily spending limits, maximum transaction size, LTV bounds, and which Morpho markets the agent can access.',
    highlight: 'policy',
    duration: 4000,
  },
  {
    id: 'policy-2',
    phase: 'policy',
    title: 'Policy Parameters',
    description: '$10,000 daily limit, $5,000 max per transaction, 70% maximum LTV, 1.2 minimum health factor. These constraints are enforced cryptographically.',
    highlight: 'policy',
    duration: 4000,
  },
  {
    id: 'policy-3',
    phase: 'policy',
    title: 'Morpho Market Whitelisting',
    description: 'The agent can only operate on approved Morpho Blue markets. USDC/WETH, USDC/wstETH, and DAI/WETH markets are whitelisted for this demo.',
    highlight: 'morpho',
    duration: 4000,
  },
  {
    id: 'auth-1',
    phase: 'authorize',
    title: 'Step 2: On-Chain Policy Registration',
    description: 'The spending policy is registered on the MorphoSpendingGate contract. This creates an immutable policy hash that the agent must prove against.',
    highlight: 'morpho',
    duration: 4000,
  },
  {
    id: 'auth-2',
    phase: 'authorize',
    title: 'Agent Authorization',
    description: 'The owner authorizes the AI agent wallet address and binds it to the registered policy. The agent cannot operate without valid zkML proofs.',
    highlight: 'policy',
    duration: 4000,
  },
  {
    id: 'agent-1',
    phase: 'agent',
    title: 'Step 3: Autonomous Agent Operations',
    description: 'The AI agent analyzes Morpho markets in real-time, looking for yield opportunities while respecting the policy constraints.',
    highlight: 'morpho',
    duration: 4000,
  },
  {
    id: 'agent-2',
    phase: 'agent',
    title: 'Agent Decision: SUPPLY',
    description: 'The agent detects high supply APY (5.21%) in the USDC/wstETH market. It decides to deploy capital to capture yield.',
    highlight: 'morpho',
    duration: 3500,
  },
  {
    id: 'agent-3',
    phase: 'agent',
    title: 'zkML Proof Generation',
    description: 'The Jolt-Atlas prover runs the agent\'s neural network inside a SNARK circuit. This takes 4-12 seconds and produces a ~48KB proof.',
    highlight: 'zkml',
    duration: 4500,
  },
  {
    id: 'agent-4',
    phase: 'agent',
    title: 'Proof Verification & Execution',
    description: 'Proof verified on-chain by MorphoSpendingGate! Policy constraints checked. Operation executes on Morpho Blue.',
    highlight: 'proof',
    duration: 3500,
  },
  {
    id: 'agent-5',
    phase: 'agent',
    title: 'Agent Decision: BORROW',
    description: 'With supply established, the agent adds leverage. Current LTV is 45%, below the 70% target. Adding leverage to increase yield.',
    highlight: 'morpho',
    duration: 3500,
  },
  {
    id: 'agent-6',
    phase: 'agent',
    title: 'Second Proof Generation',
    description: 'Another zkML proof is generated. Every operation requires fresh proof - no blanket approvals, no trust assumptions.',
    highlight: 'zkml',
    duration: 4500,
  },
  {
    id: 'agent-7',
    phase: 'agent',
    title: 'Borrow Executed',
    description: 'Proof verified, borrow executed on Morpho Blue. $2,100 borrowed against collateral. LTV now at 65%, within policy bounds.',
    highlight: 'morpho',
    duration: 3500,
  },
  {
    id: 'complete-1',
    phase: 'complete',
    title: 'Workflow Complete: Trustless Autonomous DeFi',
    description: 'The agent executed 2 operations with cryptographic policy compliance. No blind trust required - every action was verified with zkML proofs.',
    highlight: 'proof',
    duration: 5000,
  },
  {
    id: 'complete-2',
    phase: 'complete',
    title: 'Integration Summary',
    description: 'MorphoSpendingGate wraps Morpho Blue with zkML verification. ~48KB proofs, ~200K gas verification, 4-12s generation time. Production-ready.',
    highlight: 'morpho',
    duration: 6000,
  },
];

export function GuidedDemo() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Workflow state
  const [policy, setPolicy] = useState<SpendingPolicy | null>(null);
  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [proofState, setProofState] = useState<ProofState>({ status: 'idle', progress: 0 });
  const [dailySpent, setDailySpent] = useState(0);
  const [currentDecision, setCurrentDecision] = useState<typeof AGENT_DECISIONS[0] | null>(null);
  const [policyHash, setPolicyHash] = useState('');

  // Real zkML prover state
  const [useRealProver, setUseRealProver] = useState(true);
  const [proverOnline, setProverOnline] = useState<boolean | null>(null);
  const [waitingForProof, setWaitingForProof] = useState(false);
  const [lastProofResult, setLastProofResult] = useState<{
    proofHash: string;
    generationTimeMs: number;
    proofSizeBytes: number;
    approved: boolean;
    confidence: number;
  } | null>(null);

  // Check prover health on mount
  useEffect(() => {
    checkProverHealth().then(setProverOnline);
  }, []);

  const currentStep = WORKFLOW_STEPS[currentStepIndex];
  const phase = currentStep.phase;

  // Clear timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Handle step progression
  useEffect(() => {
    if (!isPlaying) return;

    // Clear any existing timer
    if (timerRef.current) clearTimeout(timerRef.current);

    // Don't advance while waiting for real proof
    if (waitingForProof) return;

    // Execute step-specific actions
    executeStepAction(currentStep.id);

    // Set timer for next step
    timerRef.current = setTimeout(() => {
      if (currentStepIndex < WORKFLOW_STEPS.length - 1) {
        setCurrentStepIndex(prev => prev + 1);
      } else {
        setIsPlaying(false);
      }
    }, currentStep.duration);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [currentStepIndex, isPlaying, waitingForProof]);

  const executeStepAction = (stepId: string) => {
    switch (stepId) {
      case 'policy-1':
        setPolicy(DEFAULT_POLICY);
        break;
      case 'auth-1':
        setPolicyHash(generateMockProofHash());
        break;
      case 'auth-2':
        setAgent({
          address: '0x7a3b...9c4e',
          ownerAddress: '0x1f2d...8a5b',
          policyHash: policyHash || generateMockProofHash(),
          strategy: 'Moderate Leverage',
          isActive: true,
        });
        break;
      case 'agent-2':
        setCurrentDecision(AGENT_DECISIONS[0]);
        setProofState({ status: 'idle', progress: 0 });
        break;
      case 'agent-3':
        runProofAnimation();
        break;
      case 'agent-4':
        setProofState({ status: 'complete', progress: 100, proofSize: 48 * 1024, proofHash: generateMockProofHash(), gasEstimate: 198500 });
        const tx1: Transaction = {
          id: 'tx-1',
          timestamp: Date.now(),
          operation: MorphoOperation.SUPPLY,
          market: 'USDC/wstETH',
          amount: 4250,
          proofHash: generateMockProofHash(),
          txHash: generateMockTxHash(),
          status: 'success',
          gasUsed: 198500,
        };
        setTransactions([tx1]);
        setDailySpent(4250);
        break;
      case 'agent-5':
        setCurrentDecision(AGENT_DECISIONS[1]);
        setProofState({ status: 'idle', progress: 0 });
        break;
      case 'agent-6':
        runProofAnimation();
        break;
      case 'agent-7':
        setProofState({ status: 'complete', progress: 100, proofSize: 48 * 1024, proofHash: generateMockProofHash(), gasEstimate: 215000 });
        const tx2: Transaction = {
          id: 'tx-2',
          timestamp: Date.now(),
          operation: MorphoOperation.BORROW,
          market: 'USDC/wstETH',
          amount: 2100,
          proofHash: generateMockProofHash(),
          txHash: generateMockTxHash(),
          status: 'success',
          gasUsed: 215000,
        };
        setTransactions(prev => [tx2, ...prev]);
        setDailySpent(prev => prev + 2100);
        break;
    }
  };

  const runProofAnimation = async (operation: 'supply' | 'borrow' = 'supply', amount: number = 4250) => {
    if (useRealProver && proverOnline) {
      // Use real NovaNet zkML prover
      setWaitingForProof(true);
      setProofState({ status: 'preparing', progress: 0 });

      try {
        const result = await generateZkmlProof(
          {
            operation,
            amountUsdc: amount,
            dailyLimitUsdc: policy?.dailyLimit || 10000,
            spentTodayUsdc: dailySpent,
            budgetUsdc: 50000,
            marketSuccessRate: 0.95,
          },
          (progress, statusText) => {
            const status = progress < 20 ? 'preparing' : progress < 70 ? 'generating' : progress < 85 ? 'signing' : 'verifying';
            setProofState({ status, progress });
          }
        );

        setLastProofResult({
          proofHash: result.proofHash,
          generationTimeMs: result.generationTimeMs,
          proofSizeBytes: result.proofSizeBytes,
          approved: result.approved,
          confidence: result.confidence,
        });

        setProofState({
          status: 'complete',
          progress: 100,
          proofSize: result.proofSizeBytes,
          proofHash: result.proofHash,
          gasEstimate: 198500,
        });

        // Auto-advance to next step after a brief delay
        setTimeout(() => {
          setWaitingForProof(false);
          if (currentStepIndex < WORKFLOW_STEPS.length - 1) {
            setCurrentStepIndex(prev => prev + 1);
          }
        }, 1500);
      } catch (error) {
        console.error('Proof generation failed:', error);
        setWaitingForProof(false);
        // Fall back to simulated proof
        runSimulatedProof();
      }
    } else {
      // Use simulated proof animation
      runSimulatedProof();
    }
  };

  const runSimulatedProof = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 5;
      if (progress <= 100) {
        const status = progress < 20 ? 'preparing' : progress < 70 ? 'generating' : progress < 85 ? 'signing' : 'verifying';
        setProofState({ status, progress });
      } else {
        clearInterval(interval);
      }
    }, 150);
  };

  const restart = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setCurrentStepIndex(0);
    setPolicy(null);
    setAgent(null);
    setTransactions([]);
    setProofState({ status: 'idle', progress: 0 });
    setDailySpent(0);
    setCurrentDecision(null);
    setPolicyHash('');
    setWaitingForProof(false);
    setLastProofResult(null);
    setIsPlaying(true);
  };

  const skipForward = () => {
    if (currentStepIndex < WORKFLOW_STEPS.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
    }
  };

  return (
    <div className="min-h-screen bg-dark-950 flex">
      {/* Left Sidebar - Annotations */}
      <div className="w-96 bg-dark-900 border-r border-dark-800 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-dark-800">
          <div className="mb-4">
            {/* NovaNet Logo */}
            <svg width="159" height="24" viewBox="0 0 106 16" fill="none" xmlns="http://www.w3.org/2000/svg" className="mb-4">
              <rect x="0.5" y="0.0820312" width="4.67097" height="14.7914" fill="url(#paint0_linear_4604_8450)"/>
              <rect x="16.0698" y="0.0820312" width="4.67097" height="14.7914" fill="url(#paint1_linear_4604_8450)"/>
              <g filter="url(#filter0_b_4604_8450)">
                <path d="M6.58957 0.0820312H0.5L6.86637 14.8735H12.9559L6.58957 0.0820312Z" fill="url(#paint2_linear_4604_8450)"/>
              </g>
              <g filter="url(#filter1_b_4604_8450)">
                <path d="M14.3747 0.0820312H8.28516L14.6515 14.8735H20.7411L14.3747 0.0820312Z" fill="url(#paint3_linear_4604_8450)"/>
              </g>
              <path d="M38.7302 0.510248V14.9199H36.7319L29.4075 4.35189H29.2738V14.9199H27.0997V0.510248H29.112L36.4435 11.0924H36.5772V0.510248H38.7302ZM45.8888 15.138C44.8756 15.138 43.9914 14.9058 43.2362 14.4415C42.481 13.9771 41.8947 13.3274 41.4772 12.4925C41.0597 11.6576 40.851 10.6819 40.851 9.56555C40.851 8.44448 41.0597 7.46414 41.4772 6.62451C41.8947 5.78488 42.481 5.13288 43.2362 4.66851C43.9914 4.20413 44.8756 3.97195 45.8888 3.97195C46.902 3.97195 47.7861 4.20413 48.5413 4.66851C49.2965 5.13288 49.8829 5.78488 50.3003 6.62451C50.7178 7.46414 50.9265 8.44448 50.9265 9.56555C50.9265 10.6819 50.7178 11.6576 50.3003 12.4925C49.8829 13.3274 49.2965 13.9771 48.5413 14.4415C47.7861 14.9058 46.902 15.138 45.8888 15.138ZM45.8958 13.372C46.5525 13.372 47.0966 13.1985 47.5282 12.8513C47.9597 12.5042 48.2787 12.0422 48.485 11.4653C48.6961 10.8883 48.8017 10.2527 48.8017 9.55851C48.8017 8.86898 48.6961 8.23575 48.485 7.6588C48.2787 7.07716 47.9597 6.61044 47.5282 6.25864C47.0966 5.90684 46.5525 5.73094 45.8958 5.73094C45.2344 5.73094 44.6856 5.90684 44.2494 6.25864C43.8178 6.61044 43.4965 7.07716 43.2855 7.6588C43.0791 8.23575 42.9759 8.86898 42.9759 9.55851C42.9759 10.2527 43.0791 10.8883 43.2855 11.4653C43.4965 12.0422 43.8178 12.5042 44.2494 12.8513C44.6856 13.1985 45.2344 13.372 45.8958 13.372ZM61.7562 4.11267L57.8371 14.9199H55.5856L51.6595 4.11267H53.9181L56.6551 12.4292H56.7677L59.4976 4.11267H61.7562ZM66.1725 15.1591C65.4877 15.1591 64.8685 15.0325 64.315 14.7792C63.7615 14.5212 63.3229 14.1483 62.9993 13.6605C62.6803 13.1727 62.5208 12.5746 62.5208 11.8663C62.5208 11.2565 62.6381 10.7546 62.8726 10.3606C63.1072 9.9666 63.4238 9.65467 63.8225 9.42483C64.2212 9.19498 64.6668 9.02143 65.1593 8.90416C65.6518 8.7869 66.1537 8.69778 66.665 8.6368C67.3123 8.56175 67.8377 8.50077 68.2411 8.45386C68.6445 8.40227 68.9376 8.32018 69.1206 8.2076C69.3035 8.09503 69.395 7.91209 69.395 7.6588V7.60955C69.395 6.99507 69.2214 6.51897 68.8743 6.18124C68.5319 5.84352 68.0206 5.67465 67.3405 5.67465C66.6322 5.67465 66.074 5.83179 65.6659 6.14606C65.2625 6.45565 64.9834 6.80041 64.8286 7.18035L62.8515 6.73005C63.086 6.07336 63.4285 5.54331 63.8788 5.13992C64.3338 4.73183 64.8568 4.43632 65.4478 4.25339C66.0388 4.06576 66.6603 3.97195 67.3123 3.97195C67.7439 3.97195 68.2012 4.02354 68.6843 4.12674C69.1722 4.22524 69.6272 4.40818 70.0493 4.67554C70.4762 4.94291 70.8256 5.3252 71.0977 5.82241C71.3697 6.31493 71.5058 6.9552 71.5058 7.74323V14.9199H69.4513V13.4424H69.3668C69.2308 13.7144 69.0268 13.9818 68.7547 14.2445C68.4826 14.5071 68.1332 14.7253 67.7063 14.8988C67.2795 15.0724 66.7682 15.1591 66.1725 15.1591ZM66.6298 13.4705C67.2115 13.4705 67.7087 13.3556 68.1215 13.1257C68.5389 12.8959 68.8555 12.5957 69.0713 12.2251C69.2918 11.8499 69.402 11.4488 69.402 11.022V9.62887C69.327 9.70392 69.1815 9.77428 68.9658 9.83995C68.7547 9.90093 68.5131 9.95487 68.2411 10.0018C67.969 10.044 67.704 10.0839 67.446 10.1214C67.188 10.1542 66.9723 10.1824 66.7987 10.2058C66.3906 10.2574 66.0177 10.3442 65.68 10.4662C65.3469 10.5881 65.0796 10.764 64.8779 10.9938C64.6809 11.219 64.5824 11.5192 64.5824 11.8945C64.5824 12.4151 64.7747 12.8091 65.1593 13.0765C65.5439 13.3392 66.0341 13.4705 66.6298 13.4705ZM85.7109 0.510248V14.9199H83.7127L76.3882 4.35189H76.2546V14.9199H74.0804V0.510248H76.0927L83.4242 11.0924H83.5579V0.510248H85.7109ZM92.9751 15.138C91.9103 15.138 90.9933 14.9105 90.224 14.4555C89.4594 13.9959 88.8684 13.3509 88.4509 12.5207C88.0381 11.6857 87.8318 10.7077 87.8318 9.58665C87.8318 8.47966 88.0381 7.50401 88.4509 6.65969C88.8684 5.81537 89.45 5.15634 90.1958 4.68258C90.9463 4.20882 91.8235 3.97195 92.8273 3.97195C93.4371 3.97195 94.0281 4.0728 94.6004 4.27449C95.1726 4.47619 95.6862 4.79281 96.1412 5.22435C96.5962 5.65589 96.9551 6.21642 97.2177 6.90595C97.4804 7.59078 97.6118 8.42337 97.6118 9.40372V10.1495H89.0208V8.57347H95.5502C95.5502 8.01998 95.4376 7.5298 95.2125 7.10296C94.9873 6.67142 94.6707 6.33134 94.2626 6.08274C93.8592 5.83413 93.3855 5.70983 92.8414 5.70983C92.2503 5.70983 91.7344 5.85524 91.2935 6.14606C90.8572 6.43219 90.5195 6.80744 90.2803 7.27182C90.0457 7.7315 89.9285 8.23106 89.9285 8.77048V10.0018C89.9285 10.7241 90.0551 11.3386 90.3084 11.8452C90.5664 12.3518 90.9252 12.7388 91.3849 13.0061C91.8446 13.2688 92.3817 13.4002 92.9962 13.4002C93.3949 13.4002 93.7584 13.3439 94.0867 13.2313C94.4151 13.114 94.6989 12.9405 94.9381 12.7106C95.1773 12.4808 95.3602 12.197 95.4869 11.8593L97.4781 12.2181C97.3186 12.8044 97.0325 13.3181 96.6197 13.759C96.2116 14.1952 95.698 14.5353 95.0788 14.7792C94.4643 15.0184 93.7631 15.138 92.9751 15.138ZM104.594 4.11267V5.8013H98.6913V4.11267H104.594ZM100.274 1.52343H102.378V11.7467C102.378 12.1548 102.439 12.462 102.561 12.6684C102.683 12.8701 102.84 13.0085 103.032 13.0835C103.229 13.1539 103.443 13.1891 103.673 13.1891C103.842 13.1891 103.989 13.1773 104.116 13.1539C104.243 13.1304 104.341 13.1117 104.412 13.0976L104.791 14.8355C104.67 14.8824 104.496 14.9293 104.271 14.9762C104.046 15.0278 103.764 15.056 103.426 15.0606C102.873 15.07 102.357 14.9715 101.879 14.7651C101.4 14.5587 101.013 14.2398 100.718 13.8082C100.422 13.3767 100.274 12.8349 100.274 12.1829V1.52343Z" fill="#D7D7E8"/>
              <defs>
                <filter id="filter0_b_4604_8450" x="-9.30581" y="-9.72378" width="32.0677" height="34.4026" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                  <feFlood floodOpacity="0" result="BackgroundImageFix"/>
                  <feGaussianBlur in="BackgroundImageFix" stdDeviation="4.90291"/>
                  <feComposite in2="SourceAlpha" operator="in" result="effect1_backgroundBlur_4604_8450"/>
                  <feBlend mode="normal" in="SourceGraphic" in2="effect1_backgroundBlur_4604_8450" result="shape"/>
                </filter>
                <filter id="filter1_b_4604_8450" x="-1.52066" y="-9.72378" width="32.0677" height="34.4026" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                  <feFlood floodOpacity="0" result="BackgroundImageFix"/>
                  <feGaussianBlur in="BackgroundImageFix" stdDeviation="4.90291"/>
                  <feComposite in2="SourceAlpha" operator="in" result="effect1_backgroundBlur_4604_8450"/>
                  <feBlend mode="normal" in="SourceGraphic" in2="effect1_backgroundBlur_4604_8450" result="shape"/>
                </filter>
                <linearGradient id="paint0_linear_4604_8450" x1="2.83549" y1="0.0820312" x2="2.83549" y2="23.854" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#383A7F"/>
                  <stop offset="1" stopColor="#8B95FF"/>
                </linearGradient>
                <linearGradient id="paint1_linear_4604_8450" x1="18.4053" y1="-5.10376" x2="18.4053" y2="14.8735" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#8B95FF"/>
                  <stop offset="1" stopColor="#383A7F"/>
                </linearGradient>
                <linearGradient id="paint2_linear_4604_8450" x1="6.72797" y1="0.0820312" x2="6.72797" y2="14.8735" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#ADB6FF"/>
                  <stop offset="1" stopColor="#5C63CB"/>
                </linearGradient>
                <linearGradient id="paint3_linear_4604_8450" x1="14.5131" y1="0.0820312" x2="14.5131" y2="14.8735" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#ADB6FF"/>
                  <stop offset="1" stopColor="#5C63CB"/>
                </linearGradient>
              </defs>
            </svg>
            {/* Tagline */}
            <p className="text-base text-dark-200 leading-relaxed mb-4">
              zkML-Verified Spending Policies for<br />
              <span className="text-morpho-400 font-semibold">Morpho Blue</span> Lending Protocol Agents
            </p>

            {/* Network Badge */}
            <a
              href={NETWORK_INFO.explorer}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-3 py-1.5 bg-dark-800 hover:bg-dark-700 rounded-lg text-xs transition-colors group"
            >
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <Globe className="w-3.5 h-3.5 text-dark-400 group-hover:text-dark-300" />
              </div>
              <span className="text-dark-300 group-hover:text-white">{NETWORK_INFO.name}</span>
              <ExternalLink className="w-3 h-3 text-dark-500" />
            </a>

            {/* Real Prover Toggle */}
            <div className="mt-4 p-3 bg-dark-800/50 rounded-lg border border-dark-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-dark-300">zkML Prover</span>
                <button
                  onClick={() => setUseRealProver(!useRealProver)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    useRealProver
                      ? 'bg-purple-500/20 text-purple-400'
                      : 'bg-dark-700 text-dark-400'
                  }`}
                >
                  {useRealProver ? <ToggleRight className="w-3.5 h-3.5" /> : <ToggleLeft className="w-3.5 h-3.5" />}
                  {useRealProver ? 'Real' : 'Simulated'}
                </button>
              </div>
              <div className="flex items-center gap-2">
                {proverOnline === null ? (
                  <div className="flex items-center gap-1.5 text-xs text-dark-500">
                    <div className="w-2 h-2 bg-dark-600 rounded-full animate-pulse" />
                    Checking...
                  </div>
                ) : proverOnline ? (
                  <div className="flex items-center gap-1.5 text-xs text-green-400">
                    <Wifi className="w-3 h-3" />
                    NovaNet Online
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 text-xs text-orange-400">
                    <WifiOff className="w-3 h-3" />
                    Offline (simulated)
                  </div>
                )}
              </div>
              {useRealProver && proverOnline && (
                <div className="mt-2 text-xs text-dark-500">
                  Proofs: ~48KB • 4-12s warm
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Current Step Annotation */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="mb-4 flex items-center justify-between">
            <span className={`px-3 py-1 rounded-full bg-gradient-to-r ${HIGHLIGHT_COLORS[currentStep.highlight]} text-white text-xs font-bold`}>
              {HIGHLIGHT_LABELS[currentStep.highlight]}
            </span>
            <span className="text-dark-500 text-sm">
              {currentStepIndex + 1} / {WORKFLOW_STEPS.length}
            </span>
          </div>

          <motion.div
            key={currentStep.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h2 className="text-xl font-bold text-white mb-3">{currentStep.title}</h2>
            <p className="text-dark-300 leading-relaxed">{currentStep.description}</p>
          </motion.div>

          {/* Progress Bar */}
          <div className="mt-6 space-y-2">
            <div className="flex gap-1">
              {WORKFLOW_STEPS.map((_, i) => (
                <div
                  key={i}
                  className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                    i < currentStepIndex
                      ? 'bg-green-500'
                      : i === currentStepIndex
                      ? `bg-gradient-to-r ${HIGHLIGHT_COLORS[currentStep.highlight]}`
                      : 'bg-dark-700'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Phase indicator */}
          <div className="mt-6 space-y-2">
            {['intro', 'policy', 'authorize', 'agent', 'complete'].map((p, i) => (
              <div
                key={p}
                className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
                  phase === p ? 'bg-dark-800' : ''
                }`}
              >
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  WORKFLOW_STEPS.findIndex(s => s.phase === p) < currentStepIndex
                    ? 'bg-green-500 text-white'
                    : phase === p
                    ? 'bg-morpho-500 text-white'
                    : 'bg-dark-700 text-dark-400'
                }`}>
                  {WORKFLOW_STEPS.findIndex(s => s.phase === p) < currentStepIndex ? '✓' : i + 1}
                </div>
                <span className={`text-sm capitalize ${phase === p ? 'text-white font-medium' : 'text-dark-500'}`}>
                  {p === 'intro' ? 'Introduction' : p === 'authorize' ? 'Authorization' : p}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer with Links and Playback Controls */}
        <div className="p-4 border-t border-dark-800">
          <div className="flex items-center justify-between">
            <div className="flex gap-4 text-xs text-dark-500">
              <a href="https://novanet.xyz" className="hover:text-purple-400">NovaNet</a>
              <a href="https://github.com/morpho-org" className="hover:text-morpho-400">Morpho</a>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={restart}
                className="p-1.5 rounded-lg bg-dark-800 hover:bg-dark-700 text-dark-300 hover:text-white transition-colors"
                title="Restart"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                  isPlaying
                    ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                    : 'bg-green-500/20 text-green-400 border border-green-500/30'
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? 'Pause' : 'Start Workflow'}
              </button>
              <button
                onClick={skipForward}
                className="p-1.5 rounded-lg bg-dark-800 hover:bg-dark-700 text-dark-300 hover:text-white transition-colors"
                title="Skip"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-8">
          <AnimatePresence mode="wait">
            {/* Intro Phase */}
            {phase === 'intro' && (
              <motion.div
                key="intro"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center min-h-[600px]"
              >
                <div className="text-center">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-morpho-500/20 to-purple-500/20 rounded-3xl mb-8"
                  >
                    <Zap className="w-12 h-12 text-morpho-400" />
                  </motion.div>
                  <h2 className="text-4xl font-bold text-white mb-4">
                    Jolt-Atlas zkML + Morpho Blue
                  </h2>
                  <p className="text-xl text-dark-400 max-w-xl mx-auto mb-12">
                    Enabling AI agents to manage DeFi positions with cryptographic policy compliance
                  </p>

                  <div className="grid grid-cols-3 gap-6 max-w-3xl mx-auto">
                    {[
                      { icon: Shield, label: 'Policy Enforcement', desc: 'Spending limits enforced on-chain' },
                      { icon: Zap, label: 'zkML Proofs', desc: '~48KB proofs in 4-12 seconds' },
                      { icon: Bot, label: 'Autonomous Agents', desc: 'AI manages positions trustlessly' },
                    ].map((item, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 + i * 0.1 }}
                        className="card p-6 text-center"
                      >
                        <item.icon className="w-8 h-8 text-morpho-400 mx-auto mb-3" />
                        <div className="font-semibold text-white mb-1">{item.label}</div>
                        <div className="text-sm text-dark-500">{item.desc}</div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Policy Phase */}
            {phase === 'policy' && policy && (
              <motion.div
                key="policy"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="mb-8">
                  <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium mb-2">
                    <Shield className="w-4 h-4" />
                    Step 1
                  </div>
                  <h2 className="text-2xl font-bold text-white">Spending Policy Configuration</h2>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Spending Limits</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-4 bg-dark-900/50 rounded-lg">
                        <span className="text-dark-400">Daily Limit</span>
                        <span className="text-2xl font-bold text-white">${policy.dailyLimit.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between items-center p-4 bg-dark-900/50 rounded-lg">
                        <span className="text-dark-400">Max Single Tx</span>
                        <span className="text-2xl font-bold text-white">${policy.maxSingleTx.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>

                  <div className="card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Risk Parameters</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-4 bg-dark-900/50 rounded-lg">
                        <span className="text-dark-400">Max LTV</span>
                        <span className="text-2xl font-bold text-white">{policy.maxLTV}%</span>
                      </div>
                      <div className="flex justify-between items-center p-4 bg-dark-900/50 rounded-lg">
                        <span className="text-dark-400">Min Health Factor</span>
                        <span className="text-2xl font-bold text-white">{policy.minHealthFactor}</span>
                      </div>
                    </div>
                  </div>

                  <div className="card p-6 col-span-2">
                    <h3 className="text-lg font-semibold text-white mb-4">Whitelisted Morpho Markets</h3>
                    <div className="grid grid-cols-3 gap-4">
                      {MOCK_MARKETS.map((market) => (
                        <div key={market.address} className="flex items-center justify-between p-4 bg-dark-900/50 rounded-lg">
                          <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-green-400" />
                            <span className="font-medium text-white">{market.name}</span>
                          </div>
                          <span className="text-green-400 font-bold">{market.supplyAPY.toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Authorize Phase */}
            {phase === 'authorize' && (
              <motion.div
                key="authorize"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="mb-8">
                  <div className="inline-flex items-center gap-2 px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-sm font-medium mb-2">
                    <Bot className="w-4 h-4" />
                    Step 2
                  </div>
                  <h2 className="text-2xl font-bold text-white">On-Chain Registration</h2>
                </div>

                <div className="max-w-2xl space-y-6">
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="card p-6"
                  >
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center">
                        <ExternalLink className="w-6 h-6 text-blue-400" />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-white">MorphoSpendingGate.registerPolicy()</div>
                        <div className="text-sm text-dark-400">Submitting policy to smart contract</div>
                      </div>
                      <CheckCircle className="w-6 h-6 text-green-400" />
                    </div>
                    {policyHash && (
                      <div className="p-3 bg-dark-900/50 rounded-lg font-mono text-sm text-dark-300 truncate">
                        Policy Hash: {policyHash}
                      </div>
                    )}
                  </motion.div>

                  {agent && (
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 }}
                      className="card p-6"
                    >
                      <div className="flex items-center gap-4 mb-4">
                        <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center">
                          <Bot className="w-6 h-6 text-purple-400" />
                        </div>
                        <div className="flex-1">
                          <div className="font-semibold text-white">MorphoSpendingGate.authorizeAgent()</div>
                          <div className="text-sm text-dark-400">Binding agent to policy constraints</div>
                        </div>
                        <CheckCircle className="w-6 h-6 text-green-400" />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 bg-dark-900/50 rounded-lg">
                          <div className="text-xs text-dark-500 mb-1">Agent Address</div>
                          <div className="font-mono text-dark-300">{agent.address}</div>
                        </div>
                        <div className="p-3 bg-dark-900/50 rounded-lg">
                          <div className="text-xs text-dark-500 mb-1">Owner Address</div>
                          <div className="font-mono text-dark-300">{agent.ownerAddress}</div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}

            {/* Agent Phase */}
            {phase === 'agent' && (
              <motion.div
                key="agent"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="mb-8">
                  <div className="inline-flex items-center gap-2 px-3 py-1 bg-morpho-500/20 text-morpho-400 rounded-full text-sm font-medium mb-2">
                    <Zap className="w-4 h-4" />
                    Step 3
                  </div>
                  <h2 className="text-2xl font-bold text-white">Autonomous Operations</h2>
                </div>

                <div className="grid grid-cols-3 gap-6">
                  <div className="col-span-2 space-y-6">
                    {/* Decision */}
                    {currentDecision && (
                      <motion.div
                        key={currentDecision.operation}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="card p-6 border-l-4 border-morpho-500"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-dark-400">Agent Decision</span>
                          <span className="text-sm text-dark-500">Confidence: {(currentDecision.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div className={`text-3xl font-bold mb-2 ${
                          currentDecision.operation === 0 ? 'text-green-400' : 'text-purple-400'
                        }`}>
                          {OPERATION_NAMES[currentDecision.operation]}
                        </div>
                        <p className="text-dark-300">{currentDecision.reasoning}</p>
                      </motion.div>
                    )}

                    {/* Proof */}
                    {proofState.status !== 'idle' && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="card p-6"
                      >
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-3">
                            <h3 className="text-lg font-semibold text-white">zkML Proof Generation</h3>
                            {useRealProver && proverOnline ? (
                              <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs font-medium rounded">
                                NovaNet Live
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 bg-dark-700 text-dark-400 text-xs font-medium rounded">
                                Simulated
                              </span>
                            )}
                          </div>
                          <div className="text-3xl font-bold text-morpho-400">{proofState.progress}%</div>
                        </div>

                        <div className="relative h-4 bg-dark-700 rounded-full mb-4 overflow-hidden">
                          <motion.div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-morpho-600 to-purple-500 rounded-full"
                            style={{ width: `${proofState.progress}%` }}
                          />
                        </div>

                        <div className="flex justify-between text-sm">
                          {['Preparing', 'Generating', 'Signing', 'Verifying'].map((stage, i) => {
                            const thresholds = [20, 70, 85, 100];
                            const isComplete = proofState.progress >= thresholds[i];
                            const isActive = proofState.progress >= (i === 0 ? 0 : thresholds[i-1]) && proofState.progress < thresholds[i];
                            return (
                              <div key={stage} className={isComplete ? 'text-green-400' : isActive ? 'text-morpho-400 font-medium' : 'text-dark-500'}>
                                {isComplete ? '✓' : isActive ? '●' : '○'} {stage}
                              </div>
                            );
                          })}
                        </div>

                        {proofState.status === 'complete' && (
                          <div className="mt-4 space-y-3">
                            <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center gap-3">
                              <CheckCircle className="w-6 h-6 text-green-400" />
                              <span className="text-green-400 font-medium">Proof verified! Executing on Morpho Blue...</span>
                            </div>
                            {lastProofResult && useRealProver && (
                              <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                                <div className="flex items-center gap-2 mb-2">
                                  <Zap className="w-4 h-4 text-purple-400" />
                                  <span className="text-sm font-medium text-purple-400">Real NovaNet Proof</span>
                                </div>
                                <div className="grid grid-cols-3 gap-3 text-sm">
                                  <div>
                                    <div className="text-dark-500">Generation</div>
                                    <div className="font-mono text-white">{(lastProofResult.generationTimeMs / 1000).toFixed(1)}s</div>
                                  </div>
                                  <div>
                                    <div className="text-dark-500">Size</div>
                                    <div className="font-mono text-white">{(lastProofResult.proofSizeBytes / 1024).toFixed(1)} KB</div>
                                  </div>
                                  <div>
                                    <div className="text-dark-500">Confidence</div>
                                    <div className="font-mono text-white">{(lastProofResult.confidence * 100).toFixed(0)}%</div>
                                  </div>
                                </div>
                                <div className="mt-2 text-xs text-dark-500 font-mono truncate">
                                  {lastProofResult.proofHash}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </motion.div>
                    )}

                    {/* Markets */}
                    <div className="card p-6">
                      <h3 className="text-lg font-semibold text-white mb-4">Morpho Markets</h3>
                      <div className="space-y-3">
                        {MOCK_MARKETS.map((market) => (
                          <div key={market.address} className="p-4 bg-dark-900/50 rounded-lg flex items-center justify-between">
                            <div>
                              <div className="font-medium text-white">{market.name}</div>
                              <div className="text-sm text-dark-500">${(market.totalSupply / 1e6).toFixed(0)}M TVL</div>
                            </div>
                            <div className="text-right">
                              <div className="text-green-400 font-bold text-lg">{market.supplyAPY.toFixed(2)}%</div>
                              <div className="text-sm text-dark-500">{market.utilization}% util</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Right Stats */}
                  <div className="space-y-6">
                    <div className="card p-6">
                      <h3 className="text-lg font-semibold text-white mb-4">Session Stats</h3>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-dark-400">Daily Spent</span>
                            <span className="text-white font-medium">${dailySpent.toLocaleString()}</span>
                          </div>
                          <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                            <div className="h-full bg-morpho-500 rounded-full transition-all" style={{ width: `${(dailySpent / 10000) * 100}%` }} />
                          </div>
                          <div className="text-xs text-dark-500 mt-1">of $10,000 limit</div>
                        </div>
                        <div className="flex justify-between py-2 border-t border-dark-700">
                          <span className="text-dark-400">Transactions</span>
                          <span className="text-white font-bold">{transactions.length}</span>
                        </div>
                        <div className="flex justify-between py-2 border-t border-dark-700">
                          <span className="text-dark-400">Proofs Generated</span>
                          <span className="text-white font-bold">{transactions.length}</span>
                        </div>
                      </div>
                    </div>

                    <div className="card p-6">
                      <h3 className="text-lg font-semibold text-white mb-4">Transactions</h3>
                      {transactions.length === 0 ? (
                        <div className="text-center py-6 text-dark-500">Waiting...</div>
                      ) : (
                        <div className="space-y-3">
                          {transactions.map((tx) => (
                            <motion.div
                              key={tx.id}
                              initial={{ opacity: 0, x: 20 }}
                              animate={{ opacity: 1, x: 0 }}
                              className="p-3 bg-dark-900/50 rounded-lg"
                            >
                              <div className="flex justify-between mb-1">
                                <span className={`font-bold ${tx.operation === 0 ? 'text-green-400' : 'text-purple-400'}`}>
                                  {OPERATION_NAMES[tx.operation]}
                                </span>
                                <span className="text-white">${tx.amount.toLocaleString()}</span>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-dark-500">
                                <CheckCircle className="w-3 h-3 text-green-400" />
                                <span className="font-mono truncate">{tx.proofHash.slice(0, 16)}...</span>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Complete Phase */}
            {phase === 'complete' && (
              <motion.div
                key="complete"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="min-h-[600px] py-8"
              >
                <div className="max-w-4xl mx-auto">
                  {/* Header */}
                  <div className="text-center mb-10">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="inline-flex items-center justify-center w-20 h-20 bg-green-500/20 rounded-2xl mb-6"
                    >
                      <CheckCircle className="w-10 h-10 text-green-400" />
                    </motion.div>
                    <h2 className="text-3xl font-bold text-white mb-3">Workflow Complete</h2>
                    <p className="text-lg text-dark-400 max-w-2xl mx-auto">
                      AI agent autonomously managed Morpho Blue positions with cryptographic policy compliance
                    </p>
                  </div>

                  {/* Stats Row */}
                  <div className="grid grid-cols-4 gap-4 mb-10">
                    {[
                      { label: 'Morpho Txns', value: '2', color: 'text-morpho-400' },
                      { label: 'zkML Proofs', value: useRealProver && proverOnline ? '2 (Real)' : '2 (Sim)', color: 'text-purple-400' },
                      { label: 'Volume', value: '$6,350', color: 'text-green-400' },
                      { label: 'Policy Violations', value: '0', color: 'text-orange-400' },
                    ].map((stat, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 * i }}
                        className="card p-4 text-center"
                      >
                        <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
                        <div className="text-sm text-dark-500">{stat.label}</div>
                      </motion.div>
                    ))}
                  </div>

                  {/* Morpho Integration Summary */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="card p-6 mb-8"
                  >
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                      <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-lg flex items-center justify-center">
                        <Zap className="w-4 h-4 text-white" />
                      </div>
                      NovaNet + Morpho Blue Integration
                    </h3>
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-semibold text-morpho-400 mb-3">What Happened</h4>
                        <ul className="space-y-2 text-sm text-dark-300">
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                            <span><strong className="text-white">MorphoSpendingGate</strong> wrapped Morpho Blue with zkML verification layer</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                            <span>Agent executed <strong className="text-white">supply()</strong> and <strong className="text-white">borrow()</strong> on whitelisted markets</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                            <span>Every Morpho call required valid <strong className="text-white">zkML spending proof</strong></span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                            <span>LTV maintained at <strong className="text-white">65%</strong> (below 70% policy max)</span>
                          </li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-purple-400 mb-3">
                          Technical Details
                          {lastProofResult && useRealProver && (
                            <span className="ml-2 px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded">Live Data</span>
                          )}
                        </h4>
                        <div className="space-y-3 text-sm">
                          <div className="flex justify-between p-2 bg-dark-900/50 rounded-lg">
                            <span className="text-dark-400">Proof Size</span>
                            <span className="text-white font-mono">
                              {lastProofResult ? `${(lastProofResult.proofSizeBytes / 1024).toFixed(1)} KB` : '~48 KB'}
                            </span>
                          </div>
                          <div className="flex justify-between p-2 bg-dark-900/50 rounded-lg">
                            <span className="text-dark-400">Generation Time</span>
                            <span className="text-white font-mono">
                              {lastProofResult ? `${(lastProofResult.generationTimeMs / 1000).toFixed(1)}s` : '4-12 sec'}
                            </span>
                          </div>
                          <div className="flex justify-between p-2 bg-dark-900/50 rounded-lg">
                            <span className="text-dark-400">Verification Gas</span>
                            <span className="text-white font-mono">~200K gas</span>
                          </div>
                          <div className="flex justify-between p-2 bg-dark-900/50 rounded-lg">
                            <span className="text-dark-400">Prover</span>
                            <span className="text-white font-mono">
                              {useRealProver && proverOnline ? 'NovaNet (Live)' : 'Simulated'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Key Takeaways */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="grid grid-cols-3 gap-4 mb-10"
                  >
                    {[
                      {
                        title: 'Morpho Markets',
                        desc: 'Agent operates only on owner-whitelisted Morpho Blue markets with verified collateral pairs',
                        color: 'from-blue-500 to-cyan-500',
                      },
                      {
                        title: 'Policy On-Chain',
                        desc: 'Spending limits, LTV bounds, and health factors enforced by MorphoSpendingGate contract',
                        color: 'from-green-500 to-emerald-500',
                      },
                      {
                        title: 'Zero Trust',
                        desc: 'No blind approvals - every Morpho interaction cryptographically proven compliant',
                        color: 'from-purple-500 to-pink-500',
                      },
                    ].map((item, i) => (
                      <div key={i} className="card p-5">
                        <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center mb-3`}>
                          <Shield className="w-5 h-5 text-white" />
                        </div>
                        <h4 className="font-semibold text-white mb-2">{item.title}</h4>
                        <p className="text-sm text-dark-400">{item.desc}</p>
                      </div>
                    ))}
                  </motion.div>

                  {/* CTA */}
                  <div className="text-center">
                    <button
                      onClick={restart}
                      className="px-8 py-3 bg-morpho-600 hover:bg-morpho-500 text-white font-semibold rounded-xl transition-colors"
                    >
                      Restart Workflow
                    </button>
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
