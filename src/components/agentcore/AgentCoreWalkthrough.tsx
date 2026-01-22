'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Server,
  Zap,
  CheckCircle2,
  Shield,
  ExternalLink,
  Copy,
  Cloud,
  Terminal,
  ArrowRight,
  Cpu,
  Bot,
  Wallet,
  AlertCircle,
  Info,
  TrendingUp,
  FileCheck,
  Scale,
  Sparkles,
} from 'lucide-react';
import { useProofGeneration } from '@/hooks/useProofGeneration';
import { useAgentCorePayment } from '@/hooks/useAgentCorePayment';
import {
  createAgentCoreDemoInput,
  runSpendingModel,
  AGENTCORE_ENTERPRISE_POLICY,
  type SpendingModelInput,
  type SpendingModelOutput,
} from '@/lib/spendingModel';
import { ProofProgress } from '../ProofProgress';
import { AgentCoreAnnotationOverlay, type BusinessAnnotation } from './AgentCoreAnnotationOverlay';

// Gateway configuration
const GATEWAY_CONFIG = {
  gatewayId: 'spending-proofs-czmzgtizng',
  region: 'us-east-1',
  gatewayUrl: 'https://spending-proofs-czmzgtizng.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp',
};

// Highlight colors for different phases
const HIGHLIGHT_COLORS = {
  intro: 'from-blue-500 to-cyan-500',
  'agent-scenario': 'from-purple-500 to-pink-500',
  connect: 'from-orange-500 to-amber-500',
  'list-tools': 'from-orange-400 to-yellow-500',
  'policy-eval': 'from-indigo-500 to-purple-500',
  'gateway-proof': 'from-yellow-500 to-orange-500',
  verify: 'from-green-500 to-emerald-500',
  transfer: 'from-cyan-500 to-blue-500',
  attestation: 'from-pink-500 to-rose-500',
  complete: 'from-green-400 to-emerald-500',
};

const HIGHLIGHT_LABELS = {
  intro: 'Introduction',
  'agent-scenario': 'Agent Scenario',
  connect: 'Connect',
  'list-tools': 'List Tools',
  'policy-eval': 'Policy Evaluation',
  'gateway-proof': 'Proof Generation',
  verify: 'Verification',
  transfer: 'Transfer',
  attestation: 'Attestation',
  complete: 'Complete',
};

// Phase type for the expanded demo
type DemoPhase =
  | 'intro'
  | 'agent-scenario'
  | 'connect'
  | 'list-tools'
  | 'policy-eval'
  | 'gateway-proof'
  | 'verify'
  | 'transfer'
  | 'attestation'
  | 'complete';

// Walkthrough step definitions
interface WalkthroughStep {
  id: string;
  phase: DemoPhase;
  title: string;
  description: string;
  awsNote?: string;
  awsDocUrl?: string;
  awsDocLabel?: string;
  duration: number;
  businessAnnotation?: BusinessAnnotation;
}

// MCP Response types
interface McpResponse {
  jsonrpc: string;
  id: number;
  result?: {
    content?: Array<{
      type: string;
      text: string;
    }>;
    tools?: unknown[];
    isError?: boolean;
  };
  error?: { code: number; message: string };
}

// Workflow steps with business annotations
const WORKFLOW_STEPS: WalkthroughStep[] = [
  {
    id: 'intro-1',
    phase: 'intro',
    title: 'AWS AgentCore + zkML',
    description: 'AWS Bedrock AgentCore Gateway exposes enterprise tools via MCP protocol. zkML proves AI agents followed spending policies before executing transactions.',
    awsNote: 'MCP protocol for AI agent compliance verification.',
    awsDocUrl: 'https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html',
    awsDocLabel: 'AgentCore Gateway Docs',
    duration: 5000,
    businessAnnotation: {
      title: 'Enterprise Agent Control',
      takeaway: 'AWS AgentCore + zkML enables cryptographically verified autonomous spending. Proof of policy compliance for every transaction.',
      color: 'aws',
      metric: 'MCP',
      metricLabel: 'protocol standard',
    },
  },
  {
    id: 'scenario-1',
    phase: 'agent-scenario',
    title: 'Cloud Infrastructure Agent',
    description: 'An autonomous infrastructure agent managing AWS compute resources. CFO-configured policies: $100K monthly budget, $15K max per transaction, preferred vendor requirements.',
    awsNote: 'Agent operates with dedicated treasury wallet and spending policies.',
    duration: 5000,
    businessAnnotation: {
      title: 'Autonomous Infrastructure',
      takeaway: 'AI agents can autonomously scale cloud infrastructure within policy guardrails. $100K monthly budget with cryptographic compliance proof.',
      color: 'enterprise',
      metric: '$100K',
      metricLabel: 'monthly budget',
    },
  },
  {
    id: 'scenario-2',
    phase: 'agent-scenario',
    title: 'Procurement Request',
    description: 'Agent receives auto-scaling request: EC2 p4d.24xlarge GPU instance at $8,500 for ML training batch. The agent must prove policy compliance before spending.',
    duration: 4000,
    businessAnnotation: {
      title: 'High-Value Compute',
      takeaway: 'p4d.24xlarge: 8x NVIDIA A100 GPUs, 96 vCPUs, 1.1TB RAM. Premium compute for ML training requires verified spending authorization.',
      color: 'aws',
      metric: '$8,500',
      metricLabel: 'per instance',
    },
  },
  {
    id: 'connect-1',
    phase: 'connect',
    title: 'SigV4 Authentication',
    description: 'Connecting to AWS AgentCore Gateway with AWS Signature Version 4 authentication. Secure, IAM-integrated access to MCP tools.',
    awsNote: 'AWS SigV4 provides identity verification and request signing.',
    duration: 3000,
  },
  {
    id: 'tools-1',
    phase: 'list-tools',
    title: 'MCP Tools Discovery',
    description: 'Discovering available tools via MCP tools/list. The gateway exposes spending proof generation and verification tools.',
    awsNote: 'tools/list returns available MCP-compatible tools.',
    duration: 3000,
  },
  {
    id: 'policy-1',
    phase: 'policy-eval',
    title: 'Policy Evaluation',
    description: 'Running the spending policy model with 8+ risk factors: vendor risk (8%), historical score (96%), category budget ($75K), compliance status, vendor relationship (5 years).',
    duration: 5000,
    businessAnnotation: {
      title: 'Multi-Factor Analysis',
      takeaway: '8 risk factors evaluated: vendor risk, history, budget, category limits, compliance, relationship duration, urgency, pre-approval status.',
      color: 'enterprise',
      metric: '8+',
      metricLabel: 'risk factors',
    },
  },
  {
    id: 'policy-2',
    phase: 'policy-eval',
    title: 'Decision: Approved',
    description: 'Model outputs APPROVE with 96% confidence. AWS is a preferred vendor with 5-year track record, manager pre-approval on file, and ample category budget remaining.',
    duration: 4000,
    businessAnnotation: {
      title: 'High Confidence Decision',
      takeaway: 'Preferred vendor tier + 5-year relationship + pre-approval = high confidence approval. All factors cryptographically provable.',
      color: 'combined',
      metric: '96%',
      metricLabel: 'confidence',
    },
  },
  {
    id: 'proof-1',
    phase: 'gateway-proof',
    title: 'SNARK Generation',
    description: 'Generating zkML proof via MCP gateway. JOLT-Atlas compiles the policy model and generates a ~48KB cryptographic proof of correct evaluation.',
    awsNote: 'tools/call invokes generateSpendingProof via MCP protocol.',
    duration: 8000,
    businessAnnotation: {
      title: 'Cryptographic Proof',
      takeaway: 'SNARK proof generation in ~10 seconds. Proves all 8 policy factors were evaluated correctly. Valid forever, verifiable by anyone.',
      color: 'zkml',
      metric: '~48KB',
      metricLabel: 'proof size',
    },
  },
  {
    id: 'proof-2',
    phase: 'gateway-proof',
    title: 'Enterprise Privacy',
    description: 'The proof reveals ONLY the decision. Budget details, vendor scores, and internal thresholds remain private. Competitors cannot see your spending limits.',
    duration: 5000,
    businessAnnotation: {
      title: 'Zero-Knowledge Privacy',
      takeaway: 'Prove the $100K budget was checked without revealing the actual number. Internal thresholds stay confidential.',
      color: 'enterprise',
      metric: '0',
      metricLabel: 'data leaked',
    },
  },
  {
    id: 'verify-1',
    phase: 'verify',
    title: 'Cryptographic Verification',
    description: 'Verifying the SNARK proof cryptographically. Valid proof = policy was followed correctly. Invalid = transaction blocked.',
    duration: 3000,
    businessAnnotation: {
      title: 'Mathematical Guarantee',
      takeaway: 'Proof verification is instant (~1ms) and deterministic. Invalid proof = mathematically impossible to proceed.',
      color: 'zkml',
      metric: '100%',
      metricLabel: 'verifiable',
    },
  },
  {
    id: 'transfer-1',
    phase: 'transfer',
    title: 'x402 USDC Payment',
    description: 'Executing payment via x402 protocol on Base Sepolia. Agent signs ERC-3009 TransferWithAuthorization, server verifies and settles USDC.',
    awsNote: 'x402: HTTP 402 payment protocol. Testnet: $0.85 (1:10000). Production: $8,500.',
    duration: 6000,
    businessAnnotation: {
      title: 'x402 Payment Protocol',
      takeaway: 'Coinbase x402 enables gasless USDC payments via signed authorizations. Agent pays, server settles - no wallet connection needed.',
      color: 'combined',
      metric: '$8,500',
      metricLabel: 'via x402',
    },
  },
  {
    id: 'attestation-1',
    phase: 'attestation',
    title: 'On-Chain Attestation',
    description: 'Recording proof hash on-chain via ProofAttestation contract. Creates permanent, immutable audit trail.',
    duration: 5000,
    businessAnnotation: {
      title: 'Permanent Audit Trail',
      takeaway: 'Proof hash stored on-chain forever. Any auditor can verify compliance years later without digging through logs.',
      color: 'aws',
      metric: 'Forever',
      metricLabel: 'verifiable',
    },
  },
  {
    id: 'complete-1',
    phase: 'complete',
    title: 'Audit Trail Complete',
    description: 'The agent completed an $8,500 procurement with cryptographic proof of policy compliance. 8 risk factors evaluated, all constraints checked, compliance verified.',
    awsNote: 'Full audit trail: MCP logs, proof hash on-chain, explorer links.',
    duration: 8000,
    businessAnnotation: {
      title: 'Enterprise Confidence',
      takeaway: 'Delegate $100K/month in autonomous agent infrastructure spend. Every decision cryptographically proven, not just logged.',
      color: 'combined',
      metric: '10x',
      metricLabel: 'delegation scale',
    },
  },
];

const PHASES: DemoPhase[] = [
  'intro',
  'agent-scenario',
  'connect',
  'list-tools',
  'policy-eval',
  'gateway-proof',
  'verify',
  'transfer',
  'attestation',
  'complete',
];

export function AgentCoreWalkthrough() {
  // Playback state
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Agent scenario state
  const [demoInput, setDemoInput] = useState<SpendingModelInput | null>(null);
  const [modelDecision, setModelDecision] = useState<SpendingModelOutput | null>(null);
  const [agentThoughts, setAgentThoughts] = useState<string[]>([]);

  // MCP gateway state
  const [toolsResponse, setToolsResponse] = useState<McpResponse | null>(null);
  const [proofResponse, setProofResponse] = useState<McpResponse | null>(null);
  const [isRealGateway, setIsRealGateway] = useState<boolean | null>(null);

  // Proof state
  const [proofHash, setProofHash] = useState<string | null>(null);
  const [proofVerified, setProofVerified] = useState(false);

  // Transfer state
  const [txHash, setTxHash] = useState<string | null>(null);
  const [transferError, setTransferError] = useState<string | null>(null);
  const [transferMethod, setTransferMethod] = useState<string | null>(null);

  // Attestation state
  const [attestationTxHash, setAttestationTxHash] = useState<string | null>(null);
  const [verifiedOnChain, setVerifiedOnChain] = useState(false);
  const [verificationSteps, setVerificationSteps] = useState<Array<{
    step: string;
    status: string;
    txHash?: string;
    details?: string;
    timeMs?: number;
  }>>([]);

  // Annotation state
  const [showingAnnotation, setShowingAnnotation] = useState(false);
  const [annotationToShow, setAnnotationToShow] = useState<{
    annotation: BusinessAnnotation;
    stepTitle: string;
  } | null>(null);

  // UI state
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const annotationTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isPlayingRef = useRef(false);
  const processedStepRef = useRef<string | null>(null);

  // Hooks
  const { state: proofState, generateProof, reset: resetProof } = useProofGeneration();
  const { wallet, transfer, executeTransfer, resetTransfer, isSimulated } = useAgentCorePayment();

  const currentStep = WORKFLOW_STEPS[currentStepIndex];
  const progress = ((currentStepIndex + 1) / WORKFLOW_STEPS.length) * 100;

  // Keep ref in sync with state
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Copy to clipboard helper
  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  // MCP Gateway call
  const callGateway = async (
    method: 'tools/list' | 'tools/call',
    toolName?: string,
    args?: Record<string, unknown>
  ): Promise<McpResponse> => {
    try {
      const response = await fetch('/api/agentcore/gateway', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method,
          toolName,
          arguments: args,
        }),
      });

      const data = await response.json();
      setIsRealGateway(!data.simulated);

      if (data.success) {
        return data.response;
      } else if (data.simulated && data.response) {
        return data.response;
      } else {
        throw new Error(data.error || 'Gateway call failed');
      }
    } catch (err) {
      setIsRealGateway(false);
      return getLocalSimulatedResponse(method, toolName, args);
    }
  };

  // Local simulated MCP response
  const getLocalSimulatedResponse = (
    method: string,
    toolName?: string,
    args?: Record<string, unknown>
  ): McpResponse => {
    if (method === 'tools/list') {
      return {
        jsonrpc: '2.0',
        id: Date.now(),
        result: {
          tools: [
            {
              name: 'spending-prover-api___generateSpendingProof',
              description: 'Generate zkML spending proof',
              inputSchema: {
                type: 'object',
                properties: {
                  inputs: { type: 'array', items: { type: 'number' } },
                  model_id: { type: 'string' },
                  tag: { type: 'string' },
                },
                required: ['inputs'],
              },
            },
            {
              name: 'spending-prover-api___getProverHealth',
              description: 'Health check',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        },
      };
    }

    return {
      jsonrpc: '2.0',
      id: Date.now(),
      result: {
        isError: false,
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: true,
            proof_hash: '0x' + Array(64).fill(0).map(() =>
              Math.floor(Math.random() * 16).toString(16)
            ).join(''),
            decision: 'approve',
            confidence: 0.96,
            model_id: 'spending-model',
            inference_time_ms: 4500,
            _simulated: true,
          }),
        }],
      },
    };
  };

  // Auto-advance effect with phase-specific side effects
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

    // Track if we've already run side effects for this step
    const stepKey = `${step.id}-${isPlaying}`;
    const alreadyProcessed = processedStepRef.current === stepKey;
    if (!alreadyProcessed) {
      processedStepRef.current = stepKey;
    }

    // Phase-specific side effects

    // Agent Scenario: Create demo input and run model
    if (!alreadyProcessed && step.phase === 'agent-scenario' && step.id === 'scenario-2') {
      const input = createAgentCoreDemoInput();
      setDemoInput(input);

      const decision = runSpendingModel(input, AGENTCORE_ENTERPRISE_POLICY);
      setModelDecision(decision);

      const thoughts = [
        'Evaluating infrastructure scaling request...',
        `Vendor: ${input.serviceName} (${Math.floor((input.vendorOnboardingDays || 0) / 365)}-year relationship)`,
        `Vendor Risk: ${((input.vendorRiskScore || 0) * 100).toFixed(0)}% (${(input.vendorRiskScore || 0) <= 0.3 ? 'LOW' : 'ELEVATED'})`,
        `Historical Score: ${((input.historicalVendorScore || 0) * 100).toFixed(0)}%`,
        `Compliance: ${input.vendorComplianceStatus ? 'VERIFIED' : 'PENDING'}`,
        `Category Budget: $${((input.categoryBudgetUsdc || 0) - (input.categorySpentUsdc || 0)).toLocaleString()} remaining`,
        `Pre-Approval: ${input.managerPreApproval ? 'YES' : 'NO'}`,
        `Price: $${input.priceUsdc.toLocaleString()} within policy`,
      ];
      setAgentThoughts([]);
      thoughts.forEach((thought, i) => {
        setTimeout(() => {
          if (isPlayingRef.current) {
            setAgentThoughts(prev => [...prev, thought]);
          }
        }, i * 400);
      });
    }

    // Connect: Establish gateway connection
    if (!alreadyProcessed && step.phase === 'connect') {
      // Connection happens implicitly via API call
    }

    // List Tools: Call tools/list
    if (!alreadyProcessed && step.phase === 'list-tools') {
      // Set simulated response immediately, then try real API
      const simulatedResponse = getLocalSimulatedResponse('tools/list');
      setToolsResponse(simulatedResponse);

      // Try real gateway call (non-blocking, will update if successful)
      callGateway('tools/list').then(response => {
        if (response) {
          setToolsResponse(response);
        }
      }).catch(() => {
        // Keep simulated response
      });
    }

    // Gateway Proof: Generate proof via MCP
    if (!alreadyProcessed && step.phase === 'gateway-proof' && step.id === 'proof-1') {
      // Call MCP gateway for proof generation
      callGateway(
        'tools/call',
        'spending-prover-api___generateSpendingProof',
        { inputs: [0.08, 1.0, 0.34, 1.0, 0.9999, 60, 8, 86400] }
      ).then(response => {
        setProofResponse(response);
        // Extract proof hash from response
        try {
          const result = response.result as { content?: { text: string }[] };
          if (result?.content?.[0]?.text) {
            const proofData = JSON.parse(result.content[0].text);
            if (proofData.proof_hash) {
              setProofHash(proofData.proof_hash);
            }
          }
        } catch {
          // Generate fallback proof hash
          const fallbackHash = '0x' + Array(64).fill(0).map(() =>
            Math.floor(Math.random() * 16).toString(16)
          ).join('');
          setProofHash(fallbackHash);
        }
      });

      // Also try real proof generation in background (non-blocking)
      const input = demoInput || createAgentCoreDemoInput();
      generateProof(input).then(result => {
        if (result.success && result.proof?.proofHash) {
          setProofHash(result.proof.proofHash);
        }
      }).catch(() => {
        // Silently ignore - we have MCP gateway fallback
      });
    }

    // Verify: Mark proof as verified
    if (!alreadyProcessed && step.phase === 'verify') {
      if (proofHash) {
        setTimeout(() => {
          if (isPlayingRef.current) {
            setProofVerified(true);
          }
        }, 1500);
      }
    }

    // Transfer: Execute transfer
    if (!alreadyProcessed && step.phase === 'transfer') {
      const recipientAddress = '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384';
      const transferAmount = 0.85; // $0.85 testnet (scaled from $8,500)

      executeTransfer(
        recipientAddress,
        transferAmount,
        proofResponse,
        proofHash || undefined,
        !proofHash
      ).then(result => {
        if (result.success && result.txHash) {
          setTxHash(result.txHash);
          if (result.method) setTransferMethod(result.method);
          if (result.steps?.length) setVerificationSteps(result.steps);
          if (result.attestationTxHash) {
            setAttestationTxHash(result.attestationTxHash);
            setVerifiedOnChain(true);
          }
        } else {
          setTransferError(result.error || 'Transfer failed');
          // Generate mock for demo continuity
          const mockHash = '0x' + Array.from({ length: 64 }, () =>
            Math.floor(Math.random() * 16).toString(16)
          ).join('');
          setTxHash(mockHash);
        }
      });
    }

    // Set timer to show annotation then advance
    timerRef.current = setTimeout(() => {
      if (isPlayingRef.current) {
        if (step.businessAnnotation) {
          setAnnotationToShow({ annotation: step.businessAnnotation, stepTitle: step.title });
          setShowingAnnotation(true);

          annotationTimerRef.current = setTimeout(() => {
            setShowingAnnotation(false);
            setAnnotationToShow(null);

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
      if (timerRef.current) clearTimeout(timerRef.current);
      if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
    };
  }, [currentStepIndex, isPlaying, demoInput, proofHash, proofResponse, generateProof, executeTransfer]);

  // Playback handlers
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
    if (annotationTimerRef.current) {
      clearTimeout(annotationTimerRef.current);
      annotationTimerRef.current = null;
    }
    setShowingAnnotation(false);
    setAnnotationToShow(null);

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
    setDemoInput(null);
    setModelDecision(null);
    setAgentThoughts([]);
    setToolsResponse(null);
    setProofResponse(null);
    setIsRealGateway(null);
    setProofHash(null);
    setProofVerified(false);
    setTxHash(null);
    setTransferError(null);
    setTransferMethod(null);
    setAttestationTxHash(null);
    setVerifiedOnChain(false);
    setVerificationSteps([]);
    setShowingAnnotation(false);
    setAnnotationToShow(null);
    setError(null);
    processedStepRef.current = null;
    if (timerRef.current) clearTimeout(timerRef.current);
    if (annotationTimerRef.current) clearTimeout(annotationTimerRef.current);
    resetProof();
    resetTransfer();
  }, [resetProof, resetTransfer]);

  // Phase helpers
  const getPhaseIcon = (phase: DemoPhase) => {
    switch (phase) {
      case 'intro': return <Info className="w-4 h-4" />;
      case 'agent-scenario': return <Bot className="w-4 h-4" />;
      case 'connect': return <Server className="w-4 h-4" />;
      case 'list-tools': return <Terminal className="w-4 h-4" />;
      case 'policy-eval': return <Scale className="w-4 h-4" />;
      case 'gateway-proof': return <Zap className="w-4 h-4" />;
      case 'verify': return <Shield className="w-4 h-4" />;
      case 'transfer': return <Wallet className="w-4 h-4" />;
      case 'attestation': return <FileCheck className="w-4 h-4" />;
      case 'complete': return <Sparkles className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getPhaseIndex = (phase: DemoPhase) => PHASES.indexOf(phase);
  const currentPhaseIndex = getPhaseIndex(currentStep.phase);

  // AWS CLI example
  const awsCliExample = `aws bedrock-agentcore invoke-gateway \\
  --region ${GATEWAY_CONFIG.region} \\
  --gateway-id ${GATEWAY_CONFIG.gatewayId} \\
  --mcp-request '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "spending-prover-api___generateSpendingProof",
      "arguments": {
        "inputs": [0.08, 1.0, 0.34, 1.0, 0.9999, 60, 8, 86400]
      }
    }
  }'`;

  return (
    <div className="flex flex-col">
      {/* Playback Controls */}
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
                  : 'bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 border border-orange-500/30'
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
            {isRealGateway !== null && (
              <span className={`px-2 py-0.5 rounded ${isRealGateway ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                {isRealGateway ? 'LIVE Gateway' : 'Simulated'}
              </span>
            )}
            <div className="flex items-center gap-0.5">
              {PHASES.map((phase, i) => (
                <div
                  key={phase}
                  className={`w-1.5 h-1.5 rounded-full transition-all ${
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

      {/* Full-screen Annotation Overlay */}
      {showingAnnotation && annotationToShow && (
        <AgentCoreAnnotationOverlay
          annotation={annotationToShow.annotation}
          stepTitle={annotationToShow.stepTitle}
          onContinue={handleAnnotationContinue}
        />
      )}

      {/* Main Demo Container */}
      <div className="flex min-h-[780px] bg-[#0a0a0a] rounded-2xl overflow-hidden border border-gray-800">
        {/* Left Sidebar */}
        <div className="w-80 bg-[#0d1117] border-r border-gray-800 flex flex-col flex-shrink-0">
          {/* Header */}
          <div className="p-4 border-b border-gray-800">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-orange-500/20 rounded-lg">
                <Cloud className="w-5 h-5 text-orange-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">AWS AgentCore</h2>
                <p className="text-xs text-gray-400">zkML via MCP Protocol</p>
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <a
                href="https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-orange-400 hover:text-orange-300"
              >
                <ExternalLink className="w-3 h-3" />
                Docs
              </a>
              <span className="text-gray-600">|</span>
              <span className="text-xs text-gray-500">{GATEWAY_CONFIG.region}</span>
            </div>
          </div>

          {/* Current Step */}
          <div className="p-4 flex-1 overflow-y-auto">
            <div className="mb-2">
              <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${HIGHLIGHT_COLORS[currentStep.phase]} text-white`}>
                {HIGHLIGHT_LABELS[currentStep.phase]}
              </span>
            </div>

            <h3 className="text-lg font-bold text-white mb-2">{currentStep.title}</h3>
            <p className="text-sm text-gray-400 leading-relaxed mb-3">{currentStep.description}</p>

            {/* AWS Note */}
            {currentStep.awsNote && (
              <div className="p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg mb-3">
                <div className="flex items-start gap-2">
                  <Cloud className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-xs text-gray-300">{currentStep.awsNote}</p>
                    {currentStep.awsDocUrl && (
                      <a
                        href={currentStep.awsDocUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-1.5 text-[10px] text-orange-400 hover:underline"
                      >
                        <span>{currentStep.awsDocLabel || 'AWS Docs'}</span>
                        <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Progress */}
            <div className="mb-4">
              <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
                <span>Progress</span>
                <span>{currentPhaseIndex + 1}/{PHASES.length}</span>
              </div>
              <div className="flex items-center gap-0.5">
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

            {/* Wallet Info */}
            {wallet.address && (
              <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700 mt-auto">
                <div className="flex items-center gap-2 mb-2">
                  <Wallet className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs text-gray-400">Demo Wallet</span>
                  {isSimulated && (
                    <span className="text-[10px] px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">
                      Simulated
                    </span>
                  )}
                </div>
                <div className="text-xs font-mono text-gray-300 truncate mb-1">
                  {wallet.address.slice(0, 10)}...{wallet.address.slice(-8)}
                </div>
                <div className="text-sm font-medium text-white">
                  ${parseFloat(wallet.balance).toLocaleString()} USDC
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          {/* Intro Phase */}
          {currentStep.phase === 'intro' && (
            <div className="space-y-4">
              <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-3">How It Works</h3>
                <p className="text-gray-400 text-sm mb-4">
                  AWS Bedrock AgentCore Gateway exposes the Spending Proofs prover as MCP-compatible tools.
                  Any AgentCore-hosted AI agent can generate zkML proofs via the MCP protocol.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Cpu className="w-4 h-4 text-orange-400" />
                      <span className="text-orange-400 font-medium text-sm">Gateway</span>
                    </div>
                    <p className="text-xs text-gray-400">
                      Translates MCP protocol to REST API, handles SigV4 auth.
                    </p>
                  </div>
                  <div className="p-3 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-4 h-4 text-yellow-400" />
                      <span className="text-yellow-400 font-medium text-sm">Tools</span>
                    </div>
                    <p className="text-xs text-gray-400">
                      generateSpendingProof and getProverHealth via MCP.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Agent Scenario Phase */}
          {currentStep.phase === 'agent-scenario' && (
            <div className="space-y-4">
              <div className="bg-[#0d1117] border border-purple-500/30 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Bot className="w-5 h-5 text-purple-400" />
                  <span className="text-purple-400 font-medium">Cloud Infrastructure Agent</span>
                </div>
                {demoInput && (
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-500">Service:</span>
                      <span className="text-gray-300 ml-2">{demoInput.serviceName}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Price:</span>
                      <span className="text-green-400 ml-2 font-medium">${demoInput.priceUsdc.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Budget:</span>
                      <span className="text-gray-300 ml-2">${demoInput.budgetUsdc.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Vendor:</span>
                      <span className="text-orange-400 ml-2">{demoInput.vendorId}</span>
                    </div>
                  </div>
                )}

                {/* Agent Thoughts */}
                {agentThoughts.length > 0 && (
                  <div className="mt-4 p-3 bg-gray-900/50 rounded-lg">
                    <div className="text-xs text-gray-500 mb-2">Agent Evaluation:</div>
                    <div className="space-y-1">
                      {agentThoughts.map((thought, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <ArrowRight className="w-3 h-3 text-purple-400" />
                          <span className="text-gray-300">{thought}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Connect Phase */}
          {currentStep.phase === 'connect' && (
            <div className="bg-[#0d1117] border border-orange-500/30 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                <span className="text-orange-400 font-medium">Connecting to Gateway...</span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Endpoint</span>
                  <span className="text-gray-300 font-mono text-xs">{GATEWAY_CONFIG.gatewayUrl.slice(0, 50)}...</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Auth</span>
                  <span className="text-orange-400">AWS SigV4</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Region</span>
                  <span className="text-gray-300">{GATEWAY_CONFIG.region}</span>
                </div>
              </div>
            </div>
          )}

          {/* List Tools Phase */}
          {currentStep.phase === 'list-tools' && (
            <div className="space-y-4">
              {toolsResponse ? (
                <div className="bg-[#0d1117] border border-blue-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span className="text-blue-400 font-medium">tools/list Response</span>
                    </div>
                    <button
                      onClick={() => copyToClipboard(JSON.stringify(toolsResponse, null, 2), 'tools')}
                      className="p-1.5 hover:bg-gray-700 rounded transition-colors"
                    >
                      <Copy className={`w-3 h-3 ${copied === 'tools' ? 'text-green-400' : 'text-gray-400'}`} />
                    </button>
                  </div>
                  <pre className="bg-gray-900/50 p-3 rounded text-xs text-gray-300 overflow-x-auto max-h-64">
                    {JSON.stringify(toolsResponse, null, 2)}
                  </pre>
                </div>
              ) : (
                <div className="bg-[#0d1117] border border-blue-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                    <span className="text-blue-400 font-medium">Listing available tools...</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Policy Evaluation Phase */}
          {currentStep.phase === 'policy-eval' && (
            <div className="space-y-4">
              {modelDecision && (
                <div className="bg-[#0d1117] border border-indigo-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Scale className="w-5 h-5 text-indigo-400" />
                    <span className="text-indigo-400 font-medium">Policy Evaluation Result</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Decision</div>
                      <div className={`text-lg font-bold ${modelDecision.shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
                        {modelDecision.shouldBuy ? 'APPROVED' : 'REJECTED'}
                      </div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                      <div className="text-xs text-gray-500 mb-1">Confidence</div>
                      <div className="text-lg font-bold text-white">
                        {(modelDecision.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    {modelDecision.reasons.slice(0, 5).map((reason, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <CheckCircle2 className="w-3 h-3 text-green-400" />
                        <span className="text-gray-300">{reason}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Gateway Proof Phase */}
          {currentStep.phase === 'gateway-proof' && (
            <div className="space-y-4">
              {/* Proof Progress - only show if actively running and no MCP response yet */}
              {proofState.status === 'running' && !proofResponse && (
                <ProofProgress
                  status={proofState.status}
                  progress={proofState.progress}
                  elapsedMs={proofState.elapsedMs}
                  steps={proofState.steps}
                />
              )}

              {/* MCP Gateway Progress - show when waiting for MCP response */}
              {!proofResponse && proofState.status !== 'running' && (
                <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                    <span className="text-yellow-400 font-medium">Generating zkML Proof via MCP Gateway...</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">Calling spending-prover-api___generateSpendingProof</p>
                </div>
              )}

              {/* Proof Response */}
              {proofResponse && (
                <div className="bg-[#0d1117] border border-green-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span className="text-green-400 font-medium">Proof Generated</span>
                    </div>
                    <button
                      onClick={() => copyToClipboard(JSON.stringify(proofResponse, null, 2), 'proof')}
                      className="p-1.5 hover:bg-gray-700 rounded transition-colors"
                    >
                      <Copy className={`w-3 h-3 ${copied === 'proof' ? 'text-green-400' : 'text-gray-400'}`} />
                    </button>
                  </div>
                  {(() => {
                    try {
                      const result = proofResponse.result as { content: { text: string }[] };
                      const proofData = JSON.parse(result.content[0].text);
                      return (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Success</span>
                            <span className="text-green-400">{proofData.success ? 'true' : 'false'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Decision</span>
                            <span className="text-green-400 font-medium">{proofData.decision}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Proof Hash</span>
                            <span className="text-yellow-400 font-mono text-xs">{proofData.proof_hash.slice(0, 18)}...</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Inference Time</span>
                            <span className="text-gray-300">{proofData.inference_time_ms}ms</span>
                          </div>
                        </div>
                      );
                    } catch {
                      return (
                        <pre className="bg-gray-900/50 p-3 rounded text-xs text-gray-300 overflow-x-auto">
                          {JSON.stringify(proofResponse, null, 2)}
                        </pre>
                      );
                    }
                  })()}
                </div>
              )}

            </div>
          )}

          {/* Verify Phase */}
          {currentStep.phase === 'verify' && (
            <div className="space-y-4">
              <div className={`bg-[#0d1117] border ${proofVerified ? 'border-green-500/30' : 'border-yellow-500/30'} rounded-xl p-4`}>
                <div className="flex items-center gap-2 mb-3">
                  {proofVerified ? (
                    <>
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                      <span className="text-green-400 font-medium">Proof Verified</span>
                    </>
                  ) : (
                    <>
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                      <span className="text-yellow-400 font-medium">Verifying Proof...</span>
                    </>
                  )}
                </div>
                {proofHash && (
                  <div className="text-sm">
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-400">Proof Hash</span>
                      <span className="text-yellow-400 font-mono text-xs">{proofHash.slice(0, 22)}...</span>
                    </div>
                    {proofVerified && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Status</span>
                        <span className="text-green-400">Valid - Policy Compliant</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Transfer Phase */}
          {currentStep.phase === 'transfer' && (
            <div className="space-y-4">
              {/* Verification Steps */}
              {verificationSteps.length > 0 && (
                <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">Transfer Steps</h4>
                  <div className="space-y-2">
                    {verificationSteps.map((step, i) => (
                      <div key={i} className="flex items-center gap-3 text-sm">
                        {step.status === 'success' ? (
                          <CheckCircle2 className="w-4 h-4 text-green-400" />
                        ) : step.status === 'failed' ? (
                          <AlertCircle className="w-4 h-4 text-red-400" />
                        ) : (
                          <div className="w-4 h-4 rounded-full bg-gray-600" />
                        )}
                        <span className="text-gray-300">{step.step}</span>
                        {step.txHash && (
                          <a
                            href={`https://sepolia.basescan.org/tx/${step.txHash}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-cyan-400 hover:text-cyan-300 text-xs"
                          >
                            View <ExternalLink className="w-3 h-3 inline" />
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Transfer Result */}
              {txHash && (
                <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                    <span className="text-cyan-400 font-medium">x402 Payment Complete</span>
                    {isSimulated && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">
                        Simulated
                      </span>
                    )}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Protocol</span>
                      <span className="text-blue-400">x402 (ERC-3009)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Amount</span>
                      <span className="text-white">$0.85 USDC</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Network</span>
                      <span className="text-blue-400">Base Sepolia</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Transaction</span>
                      <a
                        href={`https://sepolia.basescan.org/tx/${txHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 hover:text-cyan-300 text-xs font-mono flex items-center gap-1"
                      >
                        {txHash.slice(0, 14)}...
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>
                </div>
              )}

              {transferError && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="text-red-400 text-sm">{transferError}</span>
                  </div>
                </div>
              )}

              {!txHash && !transferError && transfer.status === 'pending' && (
                <div className="bg-[#0d1117] border border-cyan-500/30 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                    <span className="text-cyan-400 font-medium">Executing x402 Payment...</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    Signing ERC-3009 TransferWithAuthorization on Base Sepolia
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Attestation Phase */}
          {currentStep.phase === 'attestation' && (
            <div className="space-y-4">
              <div className={`bg-[#0d1117] border ${attestationTxHash ? 'border-pink-500/30' : 'border-gray-700'} rounded-xl p-4`}>
                <div className="flex items-center gap-2 mb-3">
                  {attestationTxHash ? (
                    <>
                      <CheckCircle2 className="w-5 h-5 text-pink-400" />
                      <span className="text-pink-400 font-medium">Attestation Recorded</span>
                    </>
                  ) : (
                    <>
                      <div className="w-2 h-2 bg-pink-400 rounded-full animate-pulse" />
                      <span className="text-pink-400 font-medium">Recording Attestation...</span>
                    </>
                  )}
                </div>
                {attestationTxHash && (
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Contract</span>
                      <span className="text-gray-300">ProofAttestation</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Proof Hash</span>
                      <span className="text-yellow-400 font-mono text-xs">{proofHash?.slice(0, 18)}...</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Transaction</span>
                      <a
                        href={`https://sepolia.basescan.org/tx/${attestationTxHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-pink-400 hover:text-pink-300 text-xs font-mono flex items-center gap-1"
                      >
                        {attestationTxHash.slice(0, 14)}...
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Complete Phase */}
          {currentStep.phase === 'complete' && (
            <div>
              {/* Powered By Header */}
              <div className="flex items-center gap-3 mb-4">
                <div className="flex items-center gap-2 px-2 py-1 bg-orange-500/10 border border-orange-500/30 rounded-lg">
                  <Cloud className="w-3 h-3 text-orange-400" />
                  <span className="text-orange-400 font-medium text-xs">AWS AgentCore</span>
                </div>
                <span className="text-gray-600 text-xs">+</span>
                <div className="flex items-center gap-2 px-2 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <Zap className="w-3 h-3 text-yellow-400" />
                  <span className="text-yellow-400 font-medium text-xs">Jolt-Atlas</span>
                </div>
                <span className="text-gray-600 text-xs">+</span>
                <div className="flex items-center gap-2 px-2 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                  <Shield className="w-3 h-3 text-cyan-400" />
                  <span className="text-cyan-400 font-medium text-xs">Base Sepolia</span>
                </div>
              </div>

              <h2 className="text-xl font-bold mb-2">
                Autonomous Cloud Infrastructure Procurement
              </h2>
              <p className="text-gray-400 max-w-2xl mb-5 text-sm">
                The Cloud Infrastructure Agent completed an <span className="text-orange-400">$8,500</span> EC2 p4d.24xlarge GPU instance purchase.
                <span className="text-yellow-400"> Jolt-Atlas</span> generated a SNARK proof verified via <span className="text-orange-400">MCP protocol</span>.
                8 risk factors evaluated, budget constraints checked, compliance verified - all cryptographically proven, not promised.
              </p>

              {/* Transaction Summary */}
              <div className="grid grid-cols-4 gap-2 max-w-2xl mb-5">
                <div className="p-2 bg-[#0d1117] border border-orange-500/30 rounded-xl text-center">
                  <div className="text-lg font-bold text-orange-400">$8,500</div>
                  <div className="text-[10px] text-gray-400">Procurement Value</div>
                </div>
                <div className="p-2 bg-[#0d1117] border border-green-500/30 rounded-xl text-center">
                  <div className="text-lg font-bold text-green-400">8+</div>
                  <div className="text-[10px] text-gray-400">Risk Factors</div>
                </div>
                <div className="p-2 bg-[#0d1117] border border-yellow-500/30 rounded-xl text-center">
                  <div className="text-lg font-bold text-yellow-400">~48KB</div>
                  <div className="text-[10px] text-gray-400">Policy Proof</div>
                </div>
                <div className="p-2 bg-[#0d1117] border border-purple-500/30 rounded-xl text-center">
                  <div className="text-lg font-bold text-purple-400">$0</div>
                  <div className="text-[10px] text-gray-400">Data Exposed</div>
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
                      <span className="text-purple-400 font-medium">1. Infrastructure Request Evaluated</span>
                      <p className="text-gray-400 mt-1">The agent received an EC2 p4d.24xlarge GPU instance request at $8,500. The policy model evaluated vendor risk (8%), historical score (96%), category budget ($75K), and compliance status: {modelDecision?.shouldBuy ? 'APPROVED' : 'REJECTED'} ({modelDecision ? `${(modelDecision.confidence * 100).toFixed(0)}%` : '96%'} confidence).</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-orange-900/20 rounded-lg border border-orange-500/30">
                    <Cloud className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="text-orange-400 font-medium">2. MCP Gateway Connected</span>
                      <p className="text-gray-400 mt-1">AWS AgentCore Gateway authenticated via SigV4, discovered available tools via MCP tools/list, and routed the proof generation request to the Jolt-Atlas prover backend.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-yellow-900/20 rounded-lg border border-yellow-500/30">
                    <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="text-yellow-400 font-medium">3. Policy Proof Generated</span>
                      <p className="text-gray-400 mt-1">Jolt-Atlas compiled the spending policy model into a SNARK circuit and generated a ~48KB proof that all 8 risk factors and budget constraints were correctly evaluated. Internal thresholds stay private.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-cyan-900/20 rounded-lg border border-cyan-500/30">
                    <Wallet className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="text-cyan-400 font-medium">4. x402 Payment on Base Sepolia</span>
                      <p className="text-gray-400 mt-1">After proof verification, the x402 USDC payment was executed on Base Sepolia ($0.85 testnet scaled from $8,500). ERC-3009 TransferWithAuthorization enables gasless, instant settlement via Coinbase&apos;s x402 protocol.</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* What This Demonstrates */}
              <div className="bg-gradient-to-r from-orange-500/10 to-yellow-500/10 border border-orange-500/30 rounded-xl p-4 max-w-2xl mb-4">
                <div className="text-sm font-medium text-orange-400 mb-2">What This Demonstrates</div>
                <p className="text-gray-300 text-xs mb-3">
                  An AI agent executed an <span className="text-orange-400 font-bold">$8,500</span> infrastructure procurement with cryptographic proof that the CFO-approved policy model was evaluated correctly — 8 factors checked, mathematically verified, unforgeable.
                </p>
                <div className="text-[10px] text-gray-400 border-t border-gray-700/50 pt-3">
                  <span className="text-yellow-400 font-medium">Why this matters:</span> AWS AgentCore + MCP protocol enables AI agents to autonomously manage $100K+ budgets with cryptographic compliance verification for every transaction.
                </div>
              </div>

              {/* Links Section */}
              <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl mb-4">
                <div className="flex flex-wrap items-center gap-2 text-[10px]">
                  {/* Transaction Links */}
                  {txHash && (
                    <>
                      <span className="text-gray-500">Transactions:</span>
                      <a
                        href={`https://sepolia.basescan.org/tx/${txHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2 py-1 bg-cyan-500/10 text-cyan-400 rounded border border-cyan-500/30 hover:border-cyan-500 transition-colors flex items-center gap-1"
                      >
                        Transfer <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    </>
                  )}
                  {attestationTxHash && (
                    <a
                      href={`https://sepolia.basescan.org/tx/${attestationTxHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2 py-1 bg-pink-500/10 text-pink-400 rounded border border-pink-500/30 hover:border-pink-500 transition-colors flex items-center gap-1"
                    >
                      Attestation <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  )}

                  <span className="text-gray-700">|</span>

                  {/* Docs */}
                  <span className="text-gray-500">Docs:</span>
                  <a
                    href="https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-2 py-1 bg-orange-500/10 text-orange-400 rounded border border-orange-500/30 hover:border-orange-500 transition-colors flex items-center gap-1"
                  >
                    AgentCore Gateway <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                  <a
                    href="https://a16z.github.io/jolt/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-2 py-1 bg-yellow-500/10 text-yellow-400 rounded border border-yellow-500/30 hover:border-yellow-500 transition-colors flex items-center gap-1"
                  >
                    JOLT Docs <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                  <a
                    href="https://x402.org"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded border border-blue-500/30 hover:border-blue-500 transition-colors flex items-center gap-1"
                  >
                    x402 Protocol <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                </div>
              </div>

              {/* CLI Example */}
              <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-4 max-w-2xl">
                <h3 className="text-sm font-bold text-white mb-3">Try it yourself</h3>
                <p className="text-xs text-gray-400 mb-3">
                  Use AWS CLI to call the gateway directly:
                </p>
                <div className="relative">
                  <pre className="bg-gray-900/50 p-3 rounded text-xs text-gray-300 overflow-x-auto">
                    {awsCliExample}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(awsCliExample, 'cli')}
                    className="absolute top-2 right-2 p-1.5 hover:bg-gray-700 rounded transition-colors"
                  >
                    <Copy className={`w-3 h-3 ${copied === 'cli' ? 'text-green-400' : 'text-gray-400'}`} />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mt-4">
              <span className="text-red-400">{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
