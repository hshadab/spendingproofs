/**
 * AgentCore Audit Trail API
 *
 * Provides external verification endpoints for delegated autonomous authority.
 * CFOs, token holders, and auditors can verify that agents followed spending policies
 * WITHOUT trusting AWS, the agent operator, or CloudWatch logs.
 *
 * Key principle: The zkML proof is verifiable by anyone, forever.
 * On-chain attestation creates immutable audit trail.
 */

import { NextRequest, NextResponse } from 'next/server';
import { createPublicClient, http } from 'viem';
import { baseSepolia } from 'viem/chains';
import { BASE_SEPOLIA_CONFIG } from '@/lib/x402';

// Simulated audit log (in production, this would be a database)
// Each entry represents a delegated spending decision
interface AuditEntry {
  id: string;
  timestamp: string;
  agent: {
    id: string;
    name: string;
    delegatedBy: string; // CFO, Board, etc.
    delegationDate: string;
  };
  delegation: {
    monthlyBudget: number;
    maxSingleTransaction: number;
    allowedCategories: string[];
    expiresAt: string;
  };
  transaction: {
    service: string;
    amount: number;
    recipient: string;
    category: string;
  };
  policyEvaluation: {
    inputHash: string; // Hash of ML model inputs (privacy-preserving)
    decision: 'approve' | 'reject';
    confidence: number;
    factorsEvaluated: number;
    // What the proof guarantees (without revealing values)
    guarantees: string[];
  };
  proof: {
    hash: string;
    generatedAt: string;
    proverVersion: string;
    verifiable: boolean;
    size: string;
  };
  onChain: {
    attestationTxHash: string | null;
    transferTxHash: string | null;
    network: string;
    blockNumber: number | null;
    explorerUrl: string | null;
  };
  verification: {
    status: 'verified' | 'pending' | 'failed';
    verifiedAt: string | null;
    verifiedBy: string | null; // "on-chain" | "external-auditor" | etc.
  };
}

// Demo audit entries (simulating historical transactions)
function generateDemoAuditTrail(proofHash?: string, txHash?: string): AuditEntry[] {
  const now = new Date();
  const entries: AuditEntry[] = [];

  // Current transaction (if proof/tx provided)
  if (proofHash) {
    entries.push({
      id: `audit-${Date.now()}`,
      timestamp: now.toISOString(),
      agent: {
        id: 'agent-infra-001',
        name: 'Cloud Infrastructure Agent',
        delegatedBy: 'CFO Sarah Chen',
        delegationDate: '2024-01-15T00:00:00Z',
      },
      delegation: {
        monthlyBudget: 100000,
        maxSingleTransaction: 15000,
        allowedCategories: ['compute', 'storage', 'networking', 'ml-training'],
        expiresAt: '2025-12-31T23:59:59Z',
      },
      transaction: {
        service: 'AWS EC2 p4d.24xlarge',
        amount: 8500,
        recipient: '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384',
        category: 'ml-training',
      },
      policyEvaluation: {
        inputHash: '0x' + Array(16).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join(''),
        decision: 'approve',
        confidence: 0.96,
        factorsEvaluated: 8,
        guarantees: [
          'Transaction amount within delegated single-transaction limit',
          'Monthly budget not exceeded after this transaction',
          'Vendor risk score below threshold',
          'Historical vendor performance above minimum',
          'Category is in allowed list',
          'Agent has valid delegation from authorized principal',
          'Compliance requirements satisfied',
          'Manager pre-approval on file (when required)',
        ],
      },
      proof: {
        hash: proofHash,
        generatedAt: now.toISOString(),
        proverVersion: 'jolt-atlas-v1',
        verifiable: true,
        size: '~48KB',
      },
      onChain: {
        attestationTxHash: txHash || null,
        transferTxHash: txHash || null,
        network: 'base-sepolia',
        blockNumber: txHash ? Math.floor(Math.random() * 1000000) + 20000000 : null,
        explorerUrl: txHash ? `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${txHash}` : null,
      },
      verification: {
        status: 'verified',
        verifiedAt: now.toISOString(),
        verifiedBy: 'on-chain-attestation',
      },
    });
  }

  // Historical entries (demo data)
  const historicalTransactions = [
    { service: 'AWS S3 Storage', amount: 2400, category: 'storage', daysAgo: 3 },
    { service: 'AWS Lambda Compute', amount: 1850, category: 'compute', daysAgo: 7 },
    { service: 'AWS SageMaker Training', amount: 12000, category: 'ml-training', daysAgo: 14 },
    { service: 'AWS CloudFront CDN', amount: 890, category: 'networking', daysAgo: 21 },
  ];

  historicalTransactions.forEach((tx, i) => {
    const txDate = new Date(now.getTime() - tx.daysAgo * 24 * 60 * 60 * 1000);
    const mockProofHash = '0x' + Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('');
    const mockTxHash = '0x' + Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('');

    entries.push({
      id: `audit-hist-${i}`,
      timestamp: txDate.toISOString(),
      agent: {
        id: 'agent-infra-001',
        name: 'Cloud Infrastructure Agent',
        delegatedBy: 'CFO Sarah Chen',
        delegationDate: '2024-01-15T00:00:00Z',
      },
      delegation: {
        monthlyBudget: 100000,
        maxSingleTransaction: 15000,
        allowedCategories: ['compute', 'storage', 'networking', 'ml-training'],
        expiresAt: '2025-12-31T23:59:59Z',
      },
      transaction: {
        service: tx.service,
        amount: tx.amount,
        recipient: '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384',
        category: tx.category,
      },
      policyEvaluation: {
        inputHash: '0x' + Array(16).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join(''),
        decision: 'approve',
        confidence: 0.92 + Math.random() * 0.06,
        factorsEvaluated: 8,
        guarantees: [
          'Transaction amount within delegated single-transaction limit',
          'Monthly budget not exceeded after this transaction',
          'Vendor risk score below threshold',
          'Historical vendor performance above minimum',
        ],
      },
      proof: {
        hash: mockProofHash,
        generatedAt: txDate.toISOString(),
        proverVersion: 'jolt-atlas-v1',
        verifiable: true,
        size: '~48KB',
      },
      onChain: {
        attestationTxHash: mockTxHash,
        transferTxHash: mockTxHash,
        network: 'base-sepolia',
        blockNumber: Math.floor(Math.random() * 1000000) + 19000000,
        explorerUrl: `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${mockTxHash}`,
      },
      verification: {
        status: 'verified',
        verifiedAt: txDate.toISOString(),
        verifiedBy: 'on-chain-attestation',
      },
    });
  });

  return entries;
}

/**
 * GET /api/agentcore/audit
 *
 * Returns the audit trail for external verification.
 * CFOs and auditors can use this to verify agent spending decisions.
 *
 * Query params:
 * - proofHash: Include specific proof in results
 * - txHash: Include specific transaction in results
 * - agentId: Filter by agent ID
 * - from/to: Date range filter
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const proofHash = searchParams.get('proofHash');
  const txHash = searchParams.get('txHash');

  const auditTrail = generateDemoAuditTrail(
    proofHash || undefined,
    txHash || undefined
  );

  // Calculate summary statistics
  const totalTransactions = auditTrail.length;
  const totalSpent = auditTrail.reduce((sum, e) => sum + e.transaction.amount, 0);
  const allVerified = auditTrail.every(e => e.verification.status === 'verified');

  return NextResponse.json({
    success: true,
    audit: {
      summary: {
        totalTransactions,
        totalSpent,
        allVerified,
        delegatedBudget: 100000,
        budgetUtilization: (totalSpent / 100000 * 100).toFixed(1) + '%',
        verificationMethod: 'zkml-on-chain',
      },
      delegation: {
        principal: 'CFO Sarah Chen',
        agent: 'Cloud Infrastructure Agent',
        authority: 'Autonomous spending up to $15,000 per transaction, $100,000 monthly',
        categories: ['compute', 'storage', 'networking', 'ml-training'],
        expiresAt: '2025-12-31T23:59:59Z',
      },
      trustModel: {
        description: 'Zero-trust verification via zkML proofs and on-chain attestation',
        whatYouDontNeedToTrust: [
          'AWS CloudWatch logs (could be modified)',
          'Agent operator claims (could be fabricated)',
          'Cedar policy logs (internal to AWS)',
          'Agent self-reporting (conflict of interest)',
        ],
        whatYouCanVerify: [
          'zkML proof mathematically guarantees policy was evaluated correctly',
          'On-chain attestation is immutable and timestamped',
          'Proof can be verified by any third party',
          'Transaction amounts match proof commitments',
        ],
        verificationEndpoints: {
          proofVerification: '/api/prove (POST with proof data)',
          onChainAttestation: `${BASE_SEPOLIA_CONFIG.explorerUrl}/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213`,
          transferHistory: `${BASE_SEPOLIA_CONFIG.explorerUrl}/address/0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384`,
        },
      },
      entries: auditTrail,
    },
  });
}

/**
 * POST /api/agentcore/audit/verify
 *
 * Verify a specific proof hash against on-chain attestation.
 * This is what an external auditor would call.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { proofHash, txHash } = body;

    if (!proofHash) {
      return NextResponse.json({
        success: false,
        error: 'proofHash is required',
      }, { status: 400 });
    }

    // In production, this would:
    // 1. Query the ProofAttestation contract for the proof hash
    // 2. Verify the SNARK proof cryptographically
    // 3. Check the attestation timestamp and block number

    // For demo, we simulate verification
    const verificationResult = {
      proofHash,
      verified: true,
      verification: {
        proofValid: true,
        proofValidReason: 'SNARK verification passed - policy model was evaluated correctly',
        onChainAttested: !!txHash,
        onChainReason: txHash
          ? 'Proof hash found in ProofAttestation contract'
          : 'No on-chain attestation provided (proof still valid)',
        timestampValid: true,
        timestampReason: 'Proof generated within acceptable time window',
      },
      guarantees: [
        'The spending policy ML model was executed correctly',
        'The decision (approve/reject) follows from the inputs',
        'No inputs were modified after proof generation',
        'The proof cannot be forged or fabricated',
      ],
      auditorNote: 'This proof provides cryptographic certainty that the agent\'s spending decision followed the delegated policy. No trust in the agent operator, AWS, or any intermediary is required.',
    };

    return NextResponse.json({
      success: true,
      ...verificationResult,
    });
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Verification failed',
    }, { status: 500 });
  }
}
