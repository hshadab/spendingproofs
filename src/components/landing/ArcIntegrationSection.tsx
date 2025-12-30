'use client';

import { Cpu, Zap, Shield, Lock, ArrowRight, Check, Code, Layers, Box } from 'lucide-react';

export function ArcIntegrationSection() {
  return (
    <>
      {/* Arc Native Integration */}
      <section id="arc-integration" className="py-16 px-6 border-t border-gray-800 bg-gradient-to-b from-[#0d1117] to-[#0a0a0a]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-full px-4 py-1.5 text-sm mb-4">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span className="text-purple-400">First-Class Primitive</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Native zkML Verification on Arc
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto text-lg">
              A custom EVM precompile for Jolt-Atlas proof verification—making spending proofs
              as fundamental to Arc as USDC itself.
            </p>
          </div>

          {/* The Vision */}
          <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-500/30 rounded-2xl p-8 mb-12">
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div>
                <h3 className="text-2xl font-bold mb-4">The Vision</h3>
                <p className="text-gray-300 mb-6">
                  Arc becomes the first L1 with <strong className="text-white">native zkML verification</strong>.
                  Every agent transaction can carry a cryptographic spending proof verified at the protocol level—not
                  in expensive Solidity, but in a purpose-built precompile optimized for HyperKZG.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center">
                      <Zap className="w-4 h-4 text-green-400" />
                    </div>
                    <span className="text-gray-300">~50k gas vs 500k+ for Solidity verification</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center">
                      <Shield className="w-4 h-4 text-green-400" />
                    </div>
                    <span className="text-gray-300">Trustless—pure cryptographic verification</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center">
                      <Lock className="w-4 h-4 text-green-400" />
                    </div>
                    <span className="text-gray-300">Policy logic stays private, only proof verified</span>
                  </div>
                </div>
              </div>
              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <div className="text-xs text-gray-500 mb-3 font-mono">// Native precompile call</div>
                <pre className="text-sm font-mono text-gray-300 overflow-x-auto">
{`// Precompile at 0x0f (proposed)
address constant JOLT_VERIFIER =
  address(0x0f);

function verifySpendingProof(
  bytes calldata proof,
  bytes32 policyHash,
  bytes32 inputsHash,
  bytes32 txIntentHash
) external view returns (bool) {
  (bool success, bytes memory result) =
    JOLT_VERIFIER.staticcall(
      abi.encode(
        proof,
        policyHash,
        inputsHash,
        txIntentHash
      )
    );
  return success &&
    abi.decode(result, (bool));
}`}
                </pre>
              </div>
            </div>
          </div>

          {/* Precompile Architecture */}
          <div className="mb-12">
            <h3 className="text-xl font-semibold mb-6 text-center">Precompile Architecture</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                  <Code className="w-5 h-5 text-purple-400" />
                </div>
                <h4 className="font-semibold mb-2">HyperKZG Verifier</h4>
                <p className="text-sm text-gray-400 mb-4">
                  Native implementation of HyperKZG polynomial commitment verification over BN254.
                  Optimized pairing operations bypass EVM overhead.
                </p>
                <div className="text-xs text-gray-500 font-mono">
                  ecPairing + sumcheck in ~50k gas
                </div>
              </div>

              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                  <Layers className="w-5 h-5 text-cyan-400" />
                </div>
                <h4 className="font-semibold mb-2">Sumcheck Protocol</h4>
                <p className="text-sm text-gray-400 mb-4">
                  Batched sumcheck verification for Jolt&apos;s lookup-based architecture.
                  No quotient polynomials—just efficient field operations.
                </p>
                <div className="text-xs text-gray-500 font-mono">
                  O(log n) verification rounds
                </div>
              </div>

              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4">
                  <Box className="w-5 h-5 text-green-400" />
                </div>
                <h4 className="font-semibold mb-2">Proof Format</h4>
                <p className="text-sm text-gray-400 mb-4">
                  Compact ~48KB proofs with structured encoding. Precompile handles deserialization
                  and curve point validation natively.
                </p>
                <div className="text-xs text-gray-500 font-mono">
                  BN254 G1/G2 points + field elements
                </div>
              </div>
            </div>
          </div>

          {/* Integration Path */}
          <div className="mb-12">
            <h3 className="text-xl font-semibold mb-6 text-center">Integration Path</h3>
            <div className="relative">
              {/* Connection line */}
              <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-purple-500 via-cyan-500 to-green-500 hidden md:block" style={{ transform: 'translateY(-50%)' }} />

              <div className="grid md:grid-cols-3 gap-6 relative">
                {/* Phase 1 */}
                <div className="bg-[#0a0a0a] border border-purple-500/30 rounded-xl p-6 relative">
                  <div className="absolute -top-3 left-4 bg-purple-500 text-white text-xs font-semibold px-2 py-0.5 rounded">
                    NOW
                  </div>
                  <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                    <Check className="w-5 h-5 text-purple-400" />
                  </div>
                  <h4 className="font-semibold mb-2">Proof Attestation</h4>
                  <p className="text-sm text-gray-400 mb-4">
                    Off-chain proof generation with on-chain attestation. Proofs verified by SDK,
                    hash committed to Arc for audit trail.
                  </p>
                  <ul className="space-y-1 text-xs text-gray-500">
                    <li className="flex items-center gap-1.5">
                      <Check className="w-3 h-3 text-purple-400" />
                      Working today
                    </li>
                    <li className="flex items-center gap-1.5">
                      <Check className="w-3 h-3 text-purple-400" />
                      ~2s proof generation
                    </li>
                    <li className="flex items-center gap-1.5">
                      <Check className="w-3 h-3 text-purple-400" />
                      Attestation on Arc
                    </li>
                  </ul>
                </div>

                {/* Phase 2 */}
                <div className="bg-[#0a0a0a] border border-cyan-500/30 rounded-xl p-6 relative">
                  <div className="absolute -top-3 left-4 bg-cyan-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                    NEXT
                  </div>
                  <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                    <Code className="w-5 h-5 text-cyan-400" />
                  </div>
                  <h4 className="font-semibold mb-2">Solidity Verifier</h4>
                  <p className="text-sm text-gray-400 mb-4">
                    Full on-chain verification via Solidity smart contract. Uses standard
                    BN254 precompiles (ecAdd, ecMul, ecPairing).
                  </p>
                  <ul className="space-y-1 text-xs text-gray-500">
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-cyan-400" />
                      ~500k gas per verification
                    </li>
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-cyan-400" />
                      Fully trustless
                    </li>
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-cyan-400" />
                      EVM portable
                    </li>
                  </ul>
                </div>

                {/* Phase 3 */}
                <div className="bg-[#0a0a0a] border border-green-500/30 rounded-xl p-6 relative">
                  <div className="absolute -top-3 left-4 bg-green-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                    VISION
                  </div>
                  <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                    <Cpu className="w-5 h-5 text-green-400" />
                  </div>
                  <h4 className="font-semibold mb-2">Native Precompile</h4>
                  <p className="text-sm text-gray-400 mb-4">
                    Arc-native Jolt verifier precompile. Purpose-built for zkML with
                    optimized HyperKZG and sumcheck operations.
                  </p>
                  <ul className="space-y-1 text-xs text-gray-500">
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-green-400" />
                      ~50k gas per verification
                    </li>
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-green-400" />
                      10x cheaper than Solidity
                    </li>
                    <li className="flex items-center gap-1.5">
                      <ArrowRight className="w-3 h-3 text-green-400" />
                      Arc-exclusive advantage
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Why Arc is Uniquely Positioned */}
          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-8">
            <h3 className="text-xl font-semibold mb-6 text-center">Why Arc is Uniquely Positioned</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Cpu className="w-6 h-6 text-purple-400" />
                </div>
                <h4 className="font-medium mb-1">Purpose-Built</h4>
                <p className="text-xs text-gray-500">
                  Arc&apos;s &quot;not general-purpose&quot; philosophy means custom precompiles for specific use cases
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-cyan-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Zap className="w-6 h-6 text-cyan-400" />
                </div>
                <h4 className="font-medium mb-1">Sub-Second Finality</h4>
                <p className="text-xs text-gray-500">
                  Proof verification + transaction finality in under 1 second enables real-time agent commerce
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Shield className="w-6 h-6 text-green-400" />
                </div>
                <h4 className="font-medium mb-1">USDC Native</h4>
                <p className="text-xs text-gray-500">
                  Agents pay for verification in USDC—predictable costs, no token volatility
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-amber-500/10 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Lock className="w-6 h-6 text-amber-400" />
                </div>
                <h4 className="font-medium mb-1">Pluggable Crypto</h4>
                <p className="text-xs text-gray-500">
                  Arc&apos;s precompile architecture already supports cryptographic backends—zkML is the next layer
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Specification */}
      <section id="precompile-spec" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Precompile Technical Specification</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Proposed specification for the Arc Jolt-Atlas verifier precompile.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Format */}
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <ArrowRight className="w-5 h-5 text-purple-400" />
                Input Format
              </h3>
              <div className="space-y-4">
                <div className="bg-[#0d1117] rounded-lg p-4">
                  <div className="text-xs text-gray-500 mb-2">Precompile Address</div>
                  <code className="text-sm font-mono text-cyan-400">0x000000000000000000000000000000000000000f</code>
                </div>
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 0-31</div>
                    <div>
                      <div className="text-sm font-medium">policyHash</div>
                      <div className="text-xs text-gray-500">Keccak256 of policy model VK</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 32-63</div>
                    <div>
                      <div className="text-sm font-medium">inputsHash</div>
                      <div className="text-xs text-gray-500">Poseidon hash of private inputs</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 64-95</div>
                    <div>
                      <div className="text-sm font-medium">txIntentHash</div>
                      <div className="text-xs text-gray-500">Hash binding proof to transaction</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 96-127</div>
                    <div>
                      <div className="text-sm font-medium">decisionHash</div>
                      <div className="text-xs text-gray-500">Hash of model outputs (shouldBuy, confidence, risk)</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 128+</div>
                    <div>
                      <div className="text-sm font-medium">proof</div>
                      <div className="text-xs text-gray-500">Serialized Jolt proof (~48KB)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Output Format & Gas */}
            <div className="space-y-6">
              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <ArrowRight className="w-5 h-5 text-cyan-400 rotate-180" />
                  Output Format
                </h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-[#0d1117] rounded-lg">
                    <div className="text-xs text-gray-500 w-24 flex-shrink-0">bytes 0-31</div>
                    <div>
                      <div className="text-sm font-medium">success</div>
                      <div className="text-xs text-gray-500">0x01 if proof valid, 0x00 otherwise</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-green-400" />
                  Gas Schedule
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-[#0d1117] rounded-lg">
                    <span className="text-sm">Base cost</span>
                    <span className="font-mono text-green-400">45,000 gas</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-[#0d1117] rounded-lg">
                    <span className="text-sm">Per KB of proof</span>
                    <span className="font-mono text-green-400">100 gas</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-[#0d1117] rounded-lg border border-green-500/30">
                    <span className="text-sm font-medium">Typical total (48KB proof)</span>
                    <span className="font-mono text-green-400 font-bold">~50,000 gas</span>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-4">
                  Compare: Solidity verifier ~500,000 gas. Native precompile is 10x more efficient.
                </p>
              </div>

              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-purple-400" />
                  Error Codes
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Invalid proof format</span>
                    <code className="text-red-400">0x01</code>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pairing check failed</span>
                    <code className="text-red-400">0x02</code>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sumcheck verification failed</span>
                    <code className="text-red-400">0x03</code>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hash mismatch</span>
                    <code className="text-red-400">0x04</code>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
