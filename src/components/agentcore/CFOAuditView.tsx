'use client';

import { useState, useEffect } from 'react';
import {
  Shield,
  CheckCircle2,
  ExternalLink,
  AlertTriangle,
  FileCheck,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  TrendingUp,
  Calendar,
  DollarSign,
  Bot,
  Building2,
  RefreshCw,
} from 'lucide-react';

interface AuditEntry {
  id: string;
  timestamp: string;
  agent: {
    id: string;
    name: string;
    delegatedBy: string;
    delegationDate: string;
  };
  transaction: {
    service: string;
    amount: number;
    category: string;
  };
  policyEvaluation: {
    decision: string;
    confidence: number;
    factorsEvaluated: number;
    guarantees: string[];
  };
  proof: {
    hash: string;
    verifiable: boolean;
  };
  onChain: {
    attestationTxHash: string | null;
    explorerUrl: string | null;
  };
  verification: {
    status: string;
  };
}

interface AuditData {
  summary: {
    totalTransactions: number;
    totalSpent: number;
    allVerified: boolean;
    delegatedBudget: number;
    budgetUtilization: string;
  };
  delegation: {
    principal: string;
    agent: string;
    authority: string;
    categories: string[];
  };
  trustModel: {
    whatYouDontNeedToTrust: string[];
    whatYouCanVerify: string[];
  };
  entries: AuditEntry[];
}

interface CFOAuditViewProps {
  proofHash?: string | null;
  txHash?: string | null;
  onClose?: () => void;
}

export function CFOAuditView({ proofHash, txHash, onClose }: CFOAuditViewProps) {
  const [auditData, setAuditData] = useState<AuditData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedEntry, setSelectedEntry] = useState<AuditEntry | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    verified: boolean;
    guarantees: string[];
  } | null>(null);

  useEffect(() => {
    fetchAuditData();
  }, [proofHash, txHash]);

  const fetchAuditData = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (proofHash) params.set('proofHash', proofHash);
      if (txHash) params.set('txHash', txHash);

      const response = await fetch(`/api/agentcore/audit?${params}`);
      const data = await response.json();

      if (data.success) {
        setAuditData(data.audit);
        if (data.audit.entries.length > 0) {
          setSelectedEntry(data.audit.entries[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch audit data:', error);
    } finally {
      setLoading(false);
    }
  };

  const verifyProof = async (entry: AuditEntry) => {
    setVerifying(true);
    try {
      const response = await fetch('/api/agentcore/audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proofHash: entry.proof.hash,
          txHash: entry.onChain.attestationTxHash,
        }),
      });
      const data = await response.json();
      if (data.success) {
        setVerificationResult({
          verified: data.verified,
          guarantees: data.guarantees,
        });
      }
    } catch (error) {
      console.error('Verification failed:', error);
    } finally {
      setVerifying(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-6">
        <div className="flex items-center gap-3">
          <RefreshCw className="w-5 h-5 text-gray-400 animate-spin" />
          <span className="text-gray-400">Loading audit trail...</span>
        </div>
      </div>
    );
  }

  if (!auditData) {
    return (
      <div className="bg-[#0d1117] border border-red-500/30 rounded-xl p-6">
        <span className="text-red-400">Failed to load audit data</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Building2 className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">CFO Audit Dashboard</h3>
              <p className="text-xs text-gray-400">External verification of delegated autonomous spending</p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          )}
        </div>

        {/* Delegation Info */}
        <div className="grid grid-cols-3 gap-3 mt-4">
          <div className="p-3 bg-gray-900/50 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Delegated By</div>
            <div className="text-sm font-medium text-white">{auditData.delegation.principal}</div>
          </div>
          <div className="p-3 bg-gray-900/50 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Agent</div>
            <div className="text-sm font-medium text-white">{auditData.delegation.agent}</div>
          </div>
          <div className="p-3 bg-gray-900/50 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Authority</div>
            <div className="text-xs text-gray-300">{auditData.delegation.authority}</div>
          </div>
        </div>
      </div>

      {/* Trust Model - Key Differentiator */}
      <div className="bg-[#0d1117] border border-yellow-500/30 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Shield className="w-5 h-5 text-yellow-400" />
          <span className="text-yellow-400 font-medium">Zero-Trust Verification</span>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <EyeOff className="w-4 h-4 text-red-400" />
              <span className="text-xs text-red-400 font-medium">You Don&apos;t Need to Trust</span>
            </div>
            <ul className="space-y-1">
              {auditData.trustModel.whatYouDontNeedToTrust.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-gray-400">
                  <Lock className="w-3 h-3 text-red-400 flex-shrink-0 mt-0.5" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Eye className="w-4 h-4 text-green-400" />
              <span className="text-xs text-green-400 font-medium">You Can Independently Verify</span>
            </div>
            <ul className="space-y-1">
              {auditData.trustModel.whatYouCanVerify.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-gray-400">
                  <Unlock className="w-3 h-3 text-green-400 flex-shrink-0 mt-0.5" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 text-center">
          <div className="text-2xl font-bold text-white">{auditData.summary.totalTransactions}</div>
          <div className="text-xs text-gray-400">Transactions</div>
        </div>
        <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 text-center">
          <div className="text-2xl font-bold text-green-400">${auditData.summary.totalSpent.toLocaleString()}</div>
          <div className="text-xs text-gray-400">Total Spent</div>
        </div>
        <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 text-center">
          <div className="text-2xl font-bold text-cyan-400">{auditData.summary.budgetUtilization}</div>
          <div className="text-xs text-gray-400">Budget Used</div>
        </div>
        <div className="bg-[#0d1117] border border-gray-700 rounded-xl p-3 text-center">
          <div className="flex items-center justify-center gap-1">
            {auditData.summary.allVerified ? (
              <CheckCircle2 className="w-6 h-6 text-green-400" />
            ) : (
              <AlertTriangle className="w-6 h-6 text-yellow-400" />
            )}
          </div>
          <div className="text-xs text-gray-400">
            {auditData.summary.allVerified ? 'All Verified' : 'Pending'}
          </div>
        </div>
      </div>

      {/* Transaction List */}
      <div className="bg-[#0d1117] border border-gray-700 rounded-xl overflow-hidden">
        <div className="p-3 border-b border-gray-700">
          <h4 className="text-sm font-medium text-white">Transaction Audit Trail</h4>
        </div>
        <div className="divide-y divide-gray-800">
          {auditData.entries.map((entry) => (
            <div
              key={entry.id}
              onClick={() => setSelectedEntry(entry)}
              className={`p-3 cursor-pointer transition-colors ${
                selectedEntry?.id === entry.id
                  ? 'bg-indigo-500/10'
                  : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-1.5 rounded ${
                    entry.verification.status === 'verified'
                      ? 'bg-green-500/20'
                      : 'bg-yellow-500/20'
                  }`}>
                    {entry.verification.status === 'verified' ? (
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-yellow-400" />
                    )}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-white">{entry.transaction.service}</div>
                    <div className="text-xs text-gray-400">
                      {new Date(entry.timestamp).toLocaleDateString()} • {entry.transaction.category}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-white">
                    ${entry.transaction.amount.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500">
                    {(entry.policyEvaluation.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Entry Details */}
      {selectedEntry && (
        <div className="bg-[#0d1117] border border-indigo-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-medium text-indigo-400">Proof Details</h4>
            <button
              onClick={() => verifyProof(selectedEntry)}
              disabled={verifying}
              className="flex items-center gap-2 px-3 py-1.5 bg-indigo-500/20 text-indigo-400 rounded-lg text-xs hover:bg-indigo-500/30 transition-colors disabled:opacity-50"
            >
              {verifying ? (
                <RefreshCw className="w-3 h-3 animate-spin" />
              ) : (
                <Shield className="w-3 h-3" />
              )}
              Verify Proof
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-500 mb-1">Proof Hash</div>
              <div className="text-xs font-mono text-yellow-400 bg-gray-900/50 p-2 rounded">
                {selectedEntry.proof.hash}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-gray-500 mb-1">Decision</div>
                <div className={`text-sm font-medium ${
                  selectedEntry.policyEvaluation.decision === 'approve'
                    ? 'text-green-400'
                    : 'text-red-400'
                }`}>
                  {selectedEntry.policyEvaluation.decision.toUpperCase()}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Factors Evaluated</div>
                <div className="text-sm font-medium text-white">
                  {selectedEntry.policyEvaluation.factorsEvaluated} factors
                </div>
              </div>
            </div>

            {selectedEntry.onChain.explorerUrl && (
              <div>
                <div className="text-xs text-gray-500 mb-1">On-Chain Attestation</div>
                <a
                  href={selectedEntry.onChain.explorerUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
                >
                  View on Base Sepolia
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}

            {/* Policy Guarantees */}
            <div>
              <div className="text-xs text-gray-500 mb-2">What This Proof Guarantees</div>
              <div className="space-y-1">
                {selectedEntry.policyEvaluation.guarantees.slice(0, 4).map((guarantee, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <CheckCircle2 className="w-3 h-3 text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-300">{guarantee}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Verification Result */}
            {verificationResult && (
              <div className={`p-3 rounded-lg ${
                verificationResult.verified
                  ? 'bg-green-500/10 border border-green-500/30'
                  : 'bg-red-500/10 border border-red-500/30'
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {verificationResult.verified ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                  )}
                  <span className={`text-sm font-medium ${
                    verificationResult.verified ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {verificationResult.verified ? 'Proof Verified' : 'Verification Failed'}
                  </span>
                </div>
                {verificationResult.verified && (
                  <p className="text-xs text-gray-400">
                    This proof cryptographically guarantees the agent followed the delegated spending policy.
                    No trust in AWS, the agent operator, or any intermediary is required.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Key Message */}
      <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <FileCheck className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-sm font-medium text-green-400 mb-1">
              Delegated Authority with Cryptographic Accountability
            </div>
            <p className="text-xs text-gray-400">
              The CFO delegated $100K/month autonomous spending authority to this agent.
              Every transaction includes a zkML proof that the spending policy was evaluated correctly.
              This proof is verifiable by anyone, forever — no trust in AWS, the agent, or CloudWatch logs required.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
