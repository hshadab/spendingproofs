'use client';

import { useState } from 'react';
import { Shield, XCircle, AlertTriangle, Play, RotateCcw, Bug, RefreshCw } from 'lucide-react';
import { validatePolicyProof, isKnownModel, PolicyValidationResult } from '@/lib/policyRegistry';

interface AttackResult {
  status: 'idle' | 'running' | 'failed';
  message?: string;
  details?: string;
}

export function AttackDemos() {
  const [modelSubstitution, setModelSubstitution] = useState<AttackResult>({ status: 'idle' });
  const [policyDowngrade, setPolicyDowngrade] = useState<AttackResult>({ status: 'idle' });

  const runModelSubstitution = async () => {
    setModelSubstitution({ status: 'running' });

    // Simulate attack attempt
    await new Promise(resolve => setTimeout(resolve, 800));

    // Attacker uses a malicious permissive model
    const maliciousModelHash = '0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef';

    // Check if model is known
    const modelCheck = await isKnownModel(maliciousModelHash);

    if (!modelCheck.known) {
      setModelSubstitution({
        status: 'failed',
        message: 'Attack Blocked: Unknown Model',
        details: `PolicyRegistry rejected proof because modelHash ${maliciousModelHash.slice(0, 20)}... is not registered for any approved policy.`,
      });
    }
  };

  const runPolicyDowngrade = async () => {
    setPolicyDowngrade({ status: 'running' });

    // Simulate attack attempt
    await new Promise(resolve => setTimeout(resolve, 800));

    // Attacker uses outdated policy version
    const result = await validatePolicyProof(
      'default-spending-policy',
      '0x7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b',
      '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
      0 // Outdated version
    );

    if (!result.valid) {
      setPolicyDowngrade({
        status: 'failed',
        message: 'Attack Blocked: Version Mismatch',
        details: result.reason || 'Policy version v0 is outdated. Current version is v1.',
      });
    }
  };

  const resetAll = () => {
    setModelSubstitution({ status: 'idle' });
    setPolicyDowngrade({ status: 'idle' });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-xl font-bold mb-2 flex items-center gap-2">
            <Bug className="w-5 h-5 text-red-400" />
            Security Attack Demos
          </h3>
          <p className="text-sm text-gray-400">
            Test how the system defends against common attack vectors. All attacks should fail.
          </p>
        </div>
        <button
          onClick={resetAll}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Attack Cards */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Model Substitution Attack */}
        <div className={`bg-[#0d1117] border rounded-xl p-5 transition-all ${
          modelSubstitution.status === 'failed'
            ? 'border-red-500/50'
            : 'border-gray-800'
        }`}>
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-orange-500/10">
                <RefreshCw className="w-5 h-5 text-orange-400" />
              </div>
              <div>
                <h4 className="font-semibold text-sm">Model Substitution</h4>
                <p className="text-xs text-gray-500">Expected: BLOCKED</p>
              </div>
            </div>
            {modelSubstitution.status === 'failed' && (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
          </div>

          <p className="text-xs text-gray-400 mb-4">
            Attacker creates a permissive model that always approves, then tries to pass it off as the legitimate policy model.
          </p>

          {modelSubstitution.status === 'idle' && (
            <button
              onClick={runModelSubstitution}
              className="w-full flex items-center justify-center gap-2 bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              <Play className="w-4 h-4" />
              Attempt Attack
            </button>
          )}

          {modelSubstitution.status === 'running' && (
            <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
              <div className="w-4 h-4 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
              Attempting model substitution...
            </div>
          )}

          {modelSubstitution.status === 'failed' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-red-400">
                <Shield className="w-4 h-4" />
                {modelSubstitution.message}
              </div>
              <div className="text-xs text-gray-400 bg-red-500/10 p-2 rounded border border-red-500/20">
                {modelSubstitution.details}
              </div>
            </div>
          )}
        </div>

        {/* Policy Downgrade Attack */}
        <div className={`bg-[#0d1117] border rounded-xl p-5 transition-all ${
          policyDowngrade.status === 'failed'
            ? 'border-red-500/50'
            : 'border-gray-800'
        }`}>
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-amber-500/10">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <h4 className="font-semibold text-sm">Policy Downgrade</h4>
                <p className="text-xs text-gray-500">Expected: BLOCKED</p>
              </div>
            </div>
            {policyDowngrade.status === 'failed' && (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
          </div>

          <p className="text-xs text-gray-400 mb-4">
            Attacker uses an outdated policy version with weaker constraints to generate proofs, hoping verifier accepts old version.
          </p>

          {policyDowngrade.status === 'idle' && (
            <button
              onClick={runPolicyDowngrade}
              className="w-full flex items-center justify-center gap-2 bg-amber-600 hover:bg-amber-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              <Play className="w-4 h-4" />
              Attempt Attack
            </button>
          )}

          {policyDowngrade.status === 'running' && (
            <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
              <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
              Attempting policy downgrade...
            </div>
          )}

          {policyDowngrade.status === 'failed' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-red-400">
                <Shield className="w-4 h-4" />
                {policyDowngrade.message}
              </div>
              <div className="text-xs text-gray-400 bg-red-500/10 p-2 rounded border border-red-500/20">
                {policyDowngrade.details}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Defense Explanation */}
      <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-4">
        <h4 className="text-sm font-semibold mb-3">How Defenses Work</h4>
        <div className="grid md:grid-cols-2 gap-4 text-xs text-gray-400">
          <div>
            <h5 className="text-orange-400 font-medium mb-1">Model Substitution Defense</h5>
            <p>
              PolicyRegistry maintains a mapping of approved policyId → modelHash.
              Verifiers check that the proof&apos;s modelHash matches the registered model.
            </p>
          </div>
          <div>
            <h5 className="text-amber-400 font-medium mb-1">Policy Downgrade Defense</h5>
            <p>
              PolicyRegistry tracks policy versions. Verifiers require the latest version,
              rejecting proofs from outdated policies with weaker constraints.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
