'use client';

import { Layers, Shield, Bot, Check, ArrowRight, CheckCircle, Code, ExternalLink, Server, Rocket } from 'lucide-react';
import { CodeBlock, verifyExample } from './utils';
import { TransactionHistory } from './TransactionHistory';

export function IntegrateSection() {
  return (
    <>
      {/* Architecture Overview */}
      <section id="architecture" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/20 rounded-full px-3 py-1 text-sm text-purple-400 mb-4">
              <Layers className="w-4 h-4" />
              For Developers
            </div>
            <h2 className="text-3xl font-bold mb-4">How It Works</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Three components work together: your app with our SDK, a prover service, and Arc chain for settlement.
            </p>
          </div>

          {/* Architecture Diagram */}
          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-8 mb-8">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Your App */}
              <div className="relative">
                <div className="bg-gradient-to-br from-purple-900/30 to-purple-900/10 border border-purple-500/30 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                      <Code className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-purple-400">YOUR APP</h3>
                      <p className="text-xs text-gray-500">Agent / Wallet / dApp</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="bg-[#0d1117] rounded-lg p-3 font-mono text-xs">
                      <span className="text-gray-500">npm install</span>
                      <br />
                      <span className="text-purple-400">@hshadab/spending-proofs</span>
                    </div>
                    <div className="bg-[#0d1117] rounded-lg p-3 font-mono text-xs text-gray-400">
                      client.prove(inputs)
                    </div>
                  </div>
                </div>
                {/* Arrow */}
                <div className="hidden lg:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                  <ArrowRight className="w-6 h-6 text-gray-600" />
                </div>
              </div>

              {/* Prover Service */}
              <div className="relative">
                <div className="bg-gradient-to-br from-cyan-900/30 to-cyan-900/10 border border-cyan-500/30 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                      <Server className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-cyan-400">PROVER</h3>
                      <p className="text-xs text-gray-500">Jolt-Atlas zkML</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2 text-gray-400">
                      <Check className="w-4 h-4 text-cyan-400" />
                      <span>Runs neural network</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-400">
                      <Check className="w-4 h-4 text-cyan-400" />
                      <span>Generates SNARK proof</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-400">
                      <Check className="w-4 h-4 text-cyan-400" />
                      <span>4-12s proving time</span>
                    </div>
                  </div>
                </div>
                {/* Arrow */}
                <div className="hidden lg:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                  <ArrowRight className="w-6 h-6 text-gray-600" />
                </div>
              </div>

              {/* Arc Chain */}
              <div>
                <div className="bg-gradient-to-br from-green-900/30 to-green-900/10 border border-green-500/30 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                      <Shield className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-green-400">ARC CHAIN</h3>
                      <p className="text-xs text-gray-500">Settlement Layer</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <a href="https://docs.arc.network/arc/key-features/deterministic-finality" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-gray-400 hover:text-green-400 transition-colors">
                      <Check className="w-4 h-4 text-green-400" />
                      <span>Sub-second finality</span>
                    </a>
                    <a href="https://docs.arc.network/arc/key-features/stable-fee-design" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-gray-400 hover:text-green-400 transition-colors">
                      <Check className="w-4 h-4 text-green-400" />
                      <span>USDC as gas token</span>
                    </a>
                    <div className="flex items-center gap-2 text-gray-400">
                      <Check className="w-4 h-4 text-green-400" />
                      <span>Proof attestation</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Flow Description */}
            <div className="mt-6 pt-6 border-t border-gray-800">
              <div className="grid md:grid-cols-4 gap-4 text-center text-sm">
                <div>
                  <div className="text-purple-400 font-mono mb-1">1. PROVE</div>
                  <p className="text-gray-500">SDK calls prover with 8 spending inputs</p>
                </div>
                <div>
                  <div className="text-cyan-400 font-mono mb-1">2. VERIFY</div>
                  <p className="text-gray-500">Cryptographic verification gates the transfer</p>
                </div>
                <div>
                  <div className="text-amber-400 font-mono mb-1">3. ATTEST</div>
                  <p className="text-gray-500">Submit proof hash for transparency</p>
                </div>
                <div>
                  <div className="text-green-400 font-mono mb-1">4. TRANSFER</div>
                  <p className="text-gray-500">Execute gated USDC payment</p>
                </div>
              </div>
            </div>
          </div>

          {/* Integration Patterns */}
          <h3 className="text-xl font-semibold mb-4 text-center">Integration Patterns</h3>
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="bg-[#0a0a0a] border border-purple-500/30 rounded-xl p-6">
              <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                <Layers className="w-5 h-5 text-purple-400" />
              </div>
              <h4 className="font-semibold mb-2">Wallet Integration</h4>
              <p className="text-sm text-gray-400">
                Smart wallets embed spending proofs as authorization layer. Every agent transaction
                carries cryptographic policy compliance.
              </p>
            </div>

            <div className="bg-[#0a0a0a] border border-cyan-500/30 rounded-xl p-6">
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-5 h-5 text-cyan-400" />
              </div>
              <h4 className="font-semibold mb-2">Protocol Gating</h4>
              <p className="text-sm text-gray-400">
                DeFi protocols require spending proofs before executing trades. Insurance protocols
                verify agent behavior before paying claims.
              </p>
            </div>

            <div className="bg-[#0a0a0a] border border-amber-500/30 rounded-xl p-6">
              <div className="w-10 h-10 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
                <Bot className="w-5 h-5 text-amber-400" />
              </div>
              <h4 className="font-semibold mb-2">Agent Frameworks</h4>
              <p className="text-sm text-gray-400">
                LangChain, AutoGPT, and custom frameworks add policy proofs as middleware.
                One integration enables every agent built on that framework.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Install SDK */}
      <section id="install-sdk" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Install the SDK</h2>
            <p className="text-gray-400">Add policy proofs to your agent in minutes</p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left: Install commands */}
            <div className="space-y-6">
              <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
                    <Code className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold">TypeScript SDK</h3>
                    <p className="text-sm text-gray-400">For Node.js, React, or browser</p>
                  </div>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg px-4 py-3 font-mono text-sm mb-4">
                  <span className="text-gray-500">$ </span>
                  <span className="text-gray-300">npm install @hshadab/spending-proofs</span>
                </div>
                <div className="flex gap-3">
                  <a
                    href="https://github.com/hshadab/spendingproofs/blob/main/docs/sdk-reference.md"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
                  >
                    API Reference
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  <a
                    href="https://github.com/hshadab/spendingproofs/tree/main/sdk"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-gray-300"
                  >
                    View source
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            </div>

            {/* Right: Quick example */}
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <h3 className="font-semibold mb-4">Quick Example</h3>
              <pre className="text-sm font-mono text-gray-300 overflow-x-auto bg-[#0d1117] rounded-lg p-4">
{`import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://prover.spendingproofs.dev'
});

const result = await client.prove({
  priceUsdc: 0.05,
  budgetUsdc: 1.00,
  spentTodayUsdc: 0.20,
  dailyLimitUsdc: 0.50,
  serviceSuccessRate: 0.95,
  serviceTotalCalls: 100,
  purchasesInCategory: 5,
  timeSinceLastPurchase: 2.5,
});

if (result.decision.shouldBuy) {
  // Proceed with transaction
  console.log('Proof hash:', result.proofHash);
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Deploy to Arc */}
      <section id="deploy" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Deploy to Arc</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Contracts are already deployed on Arc Testnet. Connect and start submitting proofs.
            </p>
          </div>

          {/* Contract Addresses */}
          <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6 mb-8">
            <h3 className="font-semibold mb-4">Arc Testnet Contracts</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-[#0d1117] rounded-lg p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-gray-500">ProofAttestation</div>
                  <a
                    href="https://testnet.arcscan.app/address/0xBE9a5DF7C551324CB872584C6E5bF56799787952"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:text-purple-300"
                  >
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <code className="text-xs text-purple-400 break-all">0xBE9a...7952</code>
                <p className="text-xs text-gray-500 mt-2">Transparency layer - logs proof submissions on-chain</p>
              </div>
              <div className="bg-[#0d1117] rounded-lg p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-gray-500">SpendingGateWallet</div>
                  <a
                    href="https://testnet.arcscan.app/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-green-400 hover:text-green-300"
                  >
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <code className="text-xs text-green-400 break-all">0x6A47...d138</code>
                <p className="text-xs text-gray-500 mt-2">Enforcement layer - gated USDC transfers</p>
              </div>
              <div className="bg-[#0d1117] rounded-lg p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs text-gray-500">TestnetUSDC</div>
                  <a
                    href="https://testnet.arcscan.app/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300"
                  >
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <code className="text-xs text-cyan-400 break-all">0x1Fb6...9213</code>
                <p className="text-xs text-gray-500 mt-2">Test USDC token (6 decimals)</p>
              </div>
            </div>

            {/* Architecture Explanation */}
            <div className="mt-4 pt-4 border-t border-gray-800">
              <h4 className="text-sm font-semibold mb-3 text-gray-300">Architecture: Verification vs Attestation</h4>
              <div className="grid md:grid-cols-2 gap-4 text-xs">
                <div className="bg-[#0d1117]/50 rounded-lg p-3 border border-cyan-500/20">
                  <div className="text-cyan-400 font-semibold mb-1">Verification (Gates Transfer)</div>
                  <p className="text-gray-400">Cryptographic proof verification via /verify endpoint. This is the trust boundary - if verification fails, transfer should not proceed.</p>
                </div>
                <div className="bg-[#0d1117]/50 rounded-lg p-3 border border-purple-500/20">
                  <div className="text-purple-400 font-semibold mb-1">Attestation (Transparency)</div>
                  <p className="text-gray-400">On-chain record of proof submission. Provides audit trail and public accountability, but relies on off-chain verification.</p>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-800">
              <div className="flex flex-wrap items-center gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Chain ID:</span>
                  <span className="text-white ml-2">5042002</span>
                </div>
                <div>
                  <span className="text-gray-500">RPC:</span>
                  <span className="text-white ml-2">https://rpc.testnet.arc.network</span>
                </div>
                <a
                  href="https://testnet.arcscan.app"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  Explorer
                  <ExternalLink className="w-3 h-3" />
                </a>
                <a
                  href="https://docs.arc.network/arc/references/connect-to-arc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gray-300 flex items-center gap-1"
                >
                  Arc Docs
                  <ExternalLink className="w-3 h-3" />
                </a>
                <a
                  href="https://faucet.circle.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gray-300 flex items-center gap-1"
                >
                  Faucet
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>

          {/* Transaction History */}
          <div className="mb-8">
            <TransactionHistory />
          </div>

          {/* Verification Example */}
          <div className="grid lg:grid-cols-2 gap-8 items-start">
            <div>
              <h3 className="text-xl font-semibold mb-4">Verify Before You Trust</h3>
              <p className="text-gray-400 mb-6">
                Merchants and protocols verify agent policy compliance before
                accepting payment. No trust required—just math.
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">Compare input hashes to detect tampering</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">Check proof attestation on Arc chain</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">Verify model hash matches expected version</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-green-400" />
                  <span className="text-gray-300">Reject transactions from rogue agents instantly</span>
                </div>
              </div>
            </div>
            <div>
              <CodeBlock code={verifyExample} />
            </div>
          </div>
        </div>
      </section>

      {/* Roadmap */}
      <section id="roadmap" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Roadmap</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Building toward protocol-native zkML verification on Arc.
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {/* Now */}
            <div className="bg-[#0a0a0a] border border-green-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-green-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                NOW
              </div>
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
              <h4 className="font-semibold mb-2">Proof Attestation</h4>
              <p className="text-sm text-gray-400 mb-2">
                Off-chain proof + on-chain hash attestation
              </p>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• SDK published to npm</li>
                <li>• Arc testnet contracts live</li>
                <li>• Circle wallet integration</li>
              </ul>
            </div>

            {/* Next */}
            <div className="bg-[#0a0a0a] border border-purple-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-purple-500 text-white text-xs font-semibold px-2 py-0.5 rounded">
                NEXT
              </div>
              <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <Shield className="w-5 h-5 text-purple-400" />
              </div>
              <h4 className="font-semibold mb-2">Solidity Verifier</h4>
              <p className="text-sm text-gray-400 mb-2">
                Full on-chain SNARK verification
              </p>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• HyperKZG verifier contract</li>
                <li>• ~500k gas per verification</li>
                <li>• SpendingGate enforcement</li>
              </ul>
            </div>

            {/* Future */}
            <div className="bg-[#0a0a0a] border border-cyan-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-cyan-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                ARC NATIVE
              </div>
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <Layers className="w-5 h-5 text-cyan-400" />
              </div>
              <h4 className="font-semibold mb-2">Native Precompile</h4>
              <p className="text-sm text-gray-400 mb-2">
                Arc-native Jolt verifier
              </p>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• ~50k gas (10x cheaper)</li>
                <li>• Consensus-level security</li>
                <li>• Arc governance proposal</li>
              </ul>
            </div>

            {/* Vision */}
            <div className="bg-[#0a0a0a] border border-amber-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-amber-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                VISION
              </div>
              <div className="w-10 h-10 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <Rocket className="w-5 h-5 text-amber-400" />
              </div>
              <h4 className="font-semibold mb-2">Agent Ecosystem</h4>
              <p className="text-sm text-gray-400 mb-2">
                Full agent commerce stack
              </p>
              <ul className="text-xs text-gray-500 space-y-1">
                <li>• Multi-chain via CCTP</li>
                <li>• Agent framework SDKs</li>
                <li>• Enterprise policy templates</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
