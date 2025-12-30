'use client';

import { ArrowRight, DollarSign, Shield, TrendingUp, Clock, ShoppingCart, CheckCircle, XCircle, AlertTriangle, Check, Zap, Box, Cpu, Code, Lock, Eye } from 'lucide-react';
import { PerformanceMetrics } from '@/components/PerformanceMetrics';

export function ProofSystemSection() {
  return (
    <>
      {/* Section 1: What's Proven */}
      <section id="whats-proven" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">What Is Actually Proven</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Understanding exactly what the cryptographic proof guarantees—and what remains private.
            </p>
          </div>

          {/* Verified Statement */}
          <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-500/30 rounded-xl p-6 mb-8 text-center">
            <h3 className="text-lg font-semibold mb-3 text-purple-400">The Verified Statement</h3>
            <p className="text-lg font-mono text-white">
              &quot;Model <span className="text-cyan-400">M</span> evaluated inputs <span className="text-cyan-400">X</span> bound to txIntentHash <span className="text-cyan-400">T</span> and produced decision <span className="text-green-400">D</span>.&quot;
            </p>
            <p className="text-sm text-gray-400 mt-3">
              This is a mathematical fact, verifiable by anyone, without revealing the private inputs.
            </p>
          </div>

          {/* Public vs Private Table */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Public Fields */}
            <div id="public-signals" className="bg-[#0a0a0a] border border-green-500/30 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-green-400" />
                Public (Verifier Sees)
              </h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">policyId</div>
                    <div className="text-xs text-gray-500">Identifier for the spending policy (e.g., &quot;default-spending-policy&quot;)</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">modelHash / vkHash</div>
                    <div className="text-xs text-gray-500">Cryptographic commitments to the model and verification key</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">txIntentHash</div>
                    <div className="text-xs text-gray-500">Binds proof to specific transaction (amount, recipient, nonce, expiry)</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">decision</div>
                    <div className="text-xs text-gray-500">shouldBuy (boolean), confidence (0-100%), riskScore (0-100)</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Private Fields */}
            <div id="private-inputs" className="bg-[#0a0a0a] border border-red-500/30 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Lock className="w-5 h-5 text-red-400" />
                Private (Hidden from Verifier)
              </h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">Policy Thresholds</div>
                    <div className="text-xs text-gray-500">Daily limit, max single purchase, min success rate, etc.</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">Model Weights</div>
                    <div className="text-xs text-gray-500">Neural network parameters and decision logic</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">Private Context Inputs</div>
                    <div className="text-xs text-gray-500">Budget remaining, spending history, behavioral patterns</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">Service Relationships</div>
                    <div className="text-xs text-gray-500">Which services the agent trusts and their reputation data</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: The Model */}
      <section id="model" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">The Spending Policy Model</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              A neural network that evaluates every purchase decision against configurable constraints.
              8 inputs, 3 outputs, one cryptographic proof.
            </p>
          </div>

          {/* Input/Output Tables */}
          <div className="grid lg:grid-cols-2 gap-8 mb-12">
            {/* Inputs */}
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ArrowRight className="w-5 h-5 text-purple-400" />
                Model Inputs
              </h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <DollarSign className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">priceUsdc</div>
                    <div className="text-xs text-gray-500">Amount the agent wants to spend on this purchase</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <DollarSign className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">budgetUsdc</div>
                    <div className="text-xs text-gray-500">Remaining budget available to the agent</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <DollarSign className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">spentTodayUsdc</div>
                    <div className="text-xs text-gray-500">Total amount already spent in the current day</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <Shield className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">dailyLimitUsdc</div>
                    <div className="text-xs text-gray-500">Policy-defined maximum daily spending limit</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <TrendingUp className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">serviceSuccessRate</div>
                    <div className="text-xs text-gray-500">Historical success rate of the service (0-1)</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <TrendingUp className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">serviceTotalCalls</div>
                    <div className="text-xs text-gray-500">Total number of calls made to this service</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <ShoppingCart className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">purchasesInCategory</div>
                    <div className="text-xs text-gray-500">Number of purchases made in this category</div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                  <Clock className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="text-sm font-medium">timeSinceLastPurchase</div>
                    <div className="text-xs text-gray-500">Hours since the last purchase was made</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Outputs */}
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ArrowRight className="w-5 h-5 text-cyan-400 rotate-180" />
                Model Outputs
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-[#0d1117] rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span className="font-medium">shouldBuy</span>
                    <span className="text-xs text-gray-500 ml-auto">boolean</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    The binary decision: should the agent proceed with this purchase?
                    True if all policy constraints are satisfied.
                  </p>
                </div>
                <div className="p-4 bg-[#0d1117] rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="w-5 h-5 text-blue-400" />
                    <span className="font-medium">confidence</span>
                    <span className="text-xs text-gray-500 ml-auto">0-100%</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    How confident the model is in its decision. Higher values indicate
                    more room within policy limits; lower values signal edge cases.
                  </p>
                </div>
                <div className="p-4 bg-[#0d1117] rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-5 h-5 text-amber-400" />
                    <span className="font-medium">riskScore</span>
                    <span className="text-xs text-gray-500 ml-auto">0-100</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Composite risk assessment factoring in budget proximity, service trust,
                    and behavioral patterns. Higher scores warrant caution.
                  </p>
                </div>
              </div>

              {/* Decision Logic */}
              <div className="mt-6 pt-6 border-t border-gray-700">
                <h4 className="text-sm font-semibold mb-3 text-gray-300">Decision Logic</h4>
                <div className="space-y-2 text-sm text-gray-400">
                  <div className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Price must not exceed remaining budget</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Spending today + price must stay within daily limit</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Service success rate weighted by total calls</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Check className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <span>Behavioral patterns influence confidence scoring</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Use Case Scenarios */}
          <h3 className="text-xl font-semibold mb-6 text-center">Example Scenarios</h3>
          <div className="grid md:grid-cols-3 gap-6">
            {/* Scenario 1: Approve */}
            <div className="bg-[#0a0a0a] border border-green-500/30 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                </div>
                <span className="font-semibold text-green-400">APPROVE</span>
              </div>
              <h4 className="font-medium mb-2">Agent Buying API Credits</h4>
              <p className="text-sm text-gray-400 mb-4">
                Agent requests $0.05 for API call. Budget is $1.00, spent $0.20 today,
                daily limit $0.50. Service has 95% success rate over 100 calls.
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500">Confidence</span>
                  <span className="text-green-400 font-mono">87%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Risk Score</span>
                  <span className="text-green-400 font-mono">12</span>
                </div>
              </div>
            </div>

            {/* Scenario 2: Reject */}
            <div className="bg-[#0a0a0a] border border-red-500/30 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-red-500/10 rounded-lg flex items-center justify-center">
                  <XCircle className="w-5 h-5 text-red-400" />
                </div>
                <span className="font-semibold text-red-400">REJECT</span>
              </div>
              <h4 className="font-medium mb-2">Suspicious Large Purchase</h4>
              <p className="text-sm text-gray-400 mb-4">
                Agent requests $0.45 purchase. Already spent $0.40 today with $0.50 limit.
                New service with only 60% success rate over 5 calls.
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500">Confidence</span>
                  <span className="text-red-400 font-mono">15%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Risk Score</span>
                  <span className="text-red-400 font-mono">85</span>
                </div>
              </div>
            </div>

            {/* Scenario 3: Edge Case */}
            <div className="bg-[#0a0a0a] border border-amber-500/30 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-amber-500/10 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                </div>
                <span className="font-semibold text-amber-400">APPROVE</span>
                <span className="text-xs text-gray-500">(low confidence)</span>
              </div>
              <h4 className="font-medium mb-2">Edge Case Purchase</h4>
              <p className="text-sm text-gray-400 mb-4">
                Agent requests $0.25 with budget $0.30. Spent $0.20 of $0.50 limit.
                Trusted service (92% rate), but tight margins trigger caution.
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-500">Confidence</span>
                  <span className="text-amber-400 font-mono">34%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Risk Score</span>
                  <span className="text-amber-400 font-mono">58</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: Jolt Atlas zkML */}
      <section id="jolt-atlas" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Jolt Atlas zkML</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              JOLT Atlas eliminates the complexity of traditional zkML. No circuits, no quotient polynomials—just
              lookup tables and the sumcheck protocol, optimized for neural network inference.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">Fast Proving</h3>
              <p className="text-sm text-gray-400">
                Prover compute p50: 2.1s, p90: 3.8s. End-to-end latency includes network
                and queue overhead. Cold starts ~8s.
              </p>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Box className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">No Circuits</h3>
              <p className="text-sm text-gray-400">
                No quotient polynomials, byte decomposition, grand products, or permutation checks.
                Lookup tables handle non-linear functions like ReLU and SoftMax natively.
              </p>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Cpu className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">Sumcheck Optimized</h3>
              <p className="text-sm text-gray-400">
                Batched sumcheck protocol delivers exceptional performance for matrix-vector
                multiplication—the core operation in neural networks.
              </p>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Code className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">ONNX Native</h3>
              <p className="text-sm text-gray-400">
                Import standard ONNX models directly. No manual circuit writing or
                model conversion required—train in PyTorch, prove with JOLT.
              </p>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Lock className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">HyperKZG + BN254</h3>
              <p className="text-sm text-gray-400">
                Production-ready polynomial commitment scheme. 45ms offchain verification.
                Compact ~48KB proofs suitable for on-chain attestation.
              </p>
            </div>

            <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">Flexible Quantization</h3>
              <p className="text-sm text-gray-400">
                Lookup tables aren&apos;t fully materialized, avoiding rigid quantization constraints.
                Future support for diverse schemes and floating-point operations.
              </p>
            </div>
          </div>

          {/* Unified Performance Metrics */}
          <div className="max-w-2xl mx-auto">
            <PerformanceMetrics variant="detailed" showBenchmarkLink={true} />
          </div>
        </div>
      </section>
    </>
  );
}
