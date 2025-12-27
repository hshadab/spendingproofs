'use client';

import { useState, useEffect } from 'react';
import { Shield, CheckCircle, XCircle, AlertTriangle, Database, Hash, FileText, Clock, User } from 'lucide-react';
import {
  lookupPolicy,
  validatePolicyProof,
  getAllPolicies,
  PolicyInfo,
  PolicyValidationResult,
} from '@/lib/policyRegistry';

interface PolicyRegistryPanelProps {
  policyId?: string;
  modelHash?: string;
  vkHash?: string;
  version?: number;
  variant?: 'compact' | 'detailed' | 'validation';
}

export function PolicyRegistryPanel({
  policyId = 'default-spending-policy',
  modelHash,
  vkHash,
  version,
  variant = 'detailed',
}: PolicyRegistryPanelProps) {
  const [policies, setPolicies] = useState<PolicyInfo[]>([]);
  const [validationResult, setValidationResult] = useState<PolicyValidationResult | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<PolicyInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPolicies();
  }, []);

  useEffect(() => {
    if (policyId && modelHash && vkHash) {
      validatePolicy();
    }
  }, [policyId, modelHash, vkHash, version]);

  const loadPolicies = async () => {
    setLoading(true);
    const allPolicies = await getAllPolicies();
    setPolicies(allPolicies);

    if (policyId) {
      const result = await lookupPolicy(policyId);
      if (result.found && result.policy) {
        setSelectedPolicy(result.policy);
      }
    }
    setLoading(false);
  };

  const validatePolicy = async () => {
    if (!policyId || !modelHash || !vkHash) return;

    const result = await validatePolicyProof(policyId, modelHash, vkHash, version);
    setValidationResult(result);
  };

  if (variant === 'validation' && validationResult) {
    return (
      <div className={`bg-[#0d1117] border rounded-xl p-5 ${
        validationResult.valid
          ? 'border-green-500/50'
          : 'border-red-500/50'
      }`}>
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              validationResult.valid
                ? 'bg-green-500/10'
                : 'bg-red-500/10'
            }`}>
              {validationResult.valid ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400" />
              )}
            </div>
            <div>
              <h4 className="font-semibold text-sm">Policy Registry Check</h4>
              <p className={`text-xs ${validationResult.valid ? 'text-green-400' : 'text-red-400'}`}>
                {validationResult.valid ? 'Approved Policy' : 'Unknown/Invalid Policy'}
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Policy ID</span>
            <span className="font-mono text-purple-400">{validationResult.policyId}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Model Hash Match</span>
            {validationResult.modelHashMatch ? (
              <span className="text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Match
              </span>
            ) : (
              <span className="text-red-400 flex items-center gap-1">
                <XCircle className="w-3 h-3" /> Mismatch
              </span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-500">VK Hash Match</span>
            {validationResult.vkHashMatch ? (
              <span className="text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Match
              </span>
            ) : (
              <span className="text-red-400 flex items-center gap-1">
                <XCircle className="w-3 h-3" /> Mismatch
              </span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Policy Active</span>
            {validationResult.isActive ? (
              <span className="text-green-400">Active</span>
            ) : (
              <span className="text-amber-400">Deprecated</span>
            )}
          </div>
          {validationResult.expectedVersion !== undefined && (
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Version</span>
              <span className={`font-mono ${validationResult.versionMatch ? 'text-white' : 'text-amber-400'}`}>
                v{validationResult.expectedVersion}
              </span>
            </div>
          )}
        </div>

        {validationResult.reason && !validationResult.valid && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-xs text-red-400">{validationResult.reason}</p>
          </div>
        )}
      </div>
    );
  }

  if (variant === 'compact') {
    return (
      <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-purple-400" />
          <h4 className="font-semibold text-sm">PolicyRegistry</h4>
        </div>
        <div className="space-y-2 text-xs">
          {policies.slice(0, 3).map(policy => (
            <div
              key={policy.policyId}
              className={`flex items-center justify-between p-2 rounded ${
                policy.isActive ? 'bg-[#0a0a0a]' : 'bg-[#0a0a0a]/50'
              }`}
            >
              <span className={`font-mono ${policy.isActive ? 'text-white' : 'text-gray-500'}`}>
                {policy.policyId}
              </span>
              <span className={`text-xs ${policy.isActive ? 'text-green-400' : 'text-gray-500'}`}>
                v{policy.version}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Detailed variant
  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-400" />
          <h4 className="font-semibold">Policy Registry</h4>
        </div>
        <span className="text-xs text-gray-500">{policies.length} registered</span>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="space-y-4">
          {policies.map(policy => (
            <div
              key={policy.policyId}
              className={`p-4 rounded-lg border transition-all cursor-pointer ${
                selectedPolicy?.policyId === policy.policyId
                  ? 'border-purple-500/50 bg-purple-500/5'
                  : 'border-gray-800 bg-[#0a0a0a] hover:border-gray-700'
              } ${!policy.isActive && 'opacity-60'}`}
              onClick={() => setSelectedPolicy(policy)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  {policy.isActive ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                  )}
                  <span className="font-mono text-sm">{policy.policyId}</span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  policy.isActive
                    ? 'bg-green-500/10 text-green-400'
                    : 'bg-amber-500/10 text-amber-400'
                }`}>
                  v{policy.version}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <Hash className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-500">Model:</span>
                  <span className="font-mono text-gray-400">{policy.modelHash.slice(0, 18)}...</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-500">VK:</span>
                  <span className="font-mono text-gray-400">{policy.vkHash.slice(0, 18)}...</span>
                </div>
                <div className="flex items-center gap-2">
                  <FileText className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-500">Metadata:</span>
                  <span className="font-mono text-cyan-400">{policy.metadataURI}</span>
                </div>
                <div className="flex items-center gap-4 mt-2 pt-2 border-t border-gray-800">
                  <div className="flex items-center gap-1 text-gray-500">
                    <User className="w-3 h-3" />
                    <span>{policy.owner.slice(0, 10)}...</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <Clock className="w-3 h-3" />
                    <span>{new Date(policy.registeredAt).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Explanation */}
      <div className="mt-6 pt-4 border-t border-gray-800">
        <h5 className="text-sm font-semibold mb-2 text-gray-300">Why Registry Matters</h5>
        <p className="text-xs text-gray-500">
          The PolicyRegistry prevents <strong className="text-white">model substitution attacks</strong>.
          Without it, a malicious agent could use a permissive model that always approves,
          then claim the proof came from your policy. The registry ensures verifiers can
          check that the proof&apos;s model hash matches the registered model for that policyId.
        </p>
      </div>
    </div>
  );
}
