'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Shield, Zap, Code, Terminal, Copy, Check, ArrowRight, Box, Lock, Cpu, Bot, DollarSign, Globe, Clock, TrendingUp, ShoppingCart, AlertTriangle, CheckCircle, XCircle, Rocket, Layers, GitBranch, Gauge } from 'lucide-react';

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={copy}
      className="p-2 hover:bg-white/10 rounded transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-gray-400" />}
    </button>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <div className="relative group">
      <pre className="bg-[#0d1117] border border-gray-800 rounded-lg p-4 overflow-x-auto">
        <code className="text-sm font-mono text-gray-300">{code}</code>
      </pre>
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <CopyButton text={code} />
      </div>
    </div>
  );
}

function InstallCommand({ pkg }: { pkg: string }) {
  return (
    <div className="flex items-center gap-3 bg-[#0d1117] border border-gray-800 rounded-lg px-4 py-3 font-mono text-sm">
      <span className="text-gray-500">$</span>
      <span className="text-gray-300 flex-1">npm install {pkg}</span>
      <CopyButton text={`npm install ${pkg}`} />
    </div>
  );
}

function ModelExplorer() {
  const [inputs, setInputs] = useState({
    price: 0.05,
    budget: 1.0,
    spentToday: 0.2,
    dailyLimit: 0.5,
  });

  // Simple decision logic for demo (mimics the actual model behavior)
  const withinDailyLimit = inputs.spentToday + inputs.price <= inputs.dailyLimit;
  const withinBudget = inputs.price <= inputs.budget;
  const shouldBuy = withinDailyLimit && withinBudget;

  // Confidence based on how much room is left
  const budgetRatio = inputs.budget > 0 ? (inputs.budget - inputs.price) / inputs.budget : 0;
  const limitRatio = inputs.dailyLimit > 0 ? (inputs.dailyLimit - inputs.spentToday - inputs.price) / inputs.dailyLimit : 0;
  const confidence = shouldBuy ? Math.round(Math.min(budgetRatio, limitRatio) * 100) : Math.round((1 - Math.min(inputs.price / inputs.budget, 1)) * 30);

  // Risk based on how close to limits
  const riskScore = shouldBuy
    ? Math.round((1 - Math.min(budgetRatio, limitRatio)) * 50)
    : Math.round(70 + Math.random() * 20);

  return (
    <div className="relative">
      {/* Glow effect */}
      <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-purple-500 to-cyan-500 rounded-xl blur-lg opacity-40 animate-pulse" />
      <div className="relative bg-[#0d1117] border border-purple-500/30 rounded-xl p-5 w-full max-w-sm shadow-[0_0_30px_rgba(168,85,247,0.15)]">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-300">Spending Model Explorer</h4>
          <span className="text-xs text-gray-500">Interactive</span>
        </div>

      {/* Inputs */}
      <div className="space-y-3 mb-5">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Price (USDC)</span>
            <span className="font-mono text-purple-400">${inputs.price.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.50"
            step="0.01"
            value={inputs.price}
            onChange={(e) => setInputs({ ...inputs, price: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Budget (USDC)</span>
            <span className="font-mono text-purple-400">${inputs.budget.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="2.00"
            step="0.10"
            value={inputs.budget}
            onChange={(e) => setInputs({ ...inputs, budget: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Spent Today (USDC)</span>
            <span className="font-mono text-purple-400">${inputs.spentToday.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.00"
            max="1.00"
            step="0.05"
            value={inputs.spentToday}
            onChange={(e) => setInputs({ ...inputs, spentToday: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-500">Daily Limit (USDC)</span>
            <span className="font-mono text-purple-400">${inputs.dailyLimit.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="1.00"
            step="0.05"
            value={inputs.dailyLimit}
            onChange={(e) => setInputs({ ...inputs, dailyLimit: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-gray-700 my-4" />

      {/* Outputs */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Decision</span>
          <span className={`text-sm font-semibold ${shouldBuy ? 'text-green-400' : 'text-red-400'}`}>
            {shouldBuy ? 'APPROVE' : 'REJECT'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Confidence</span>
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${shouldBuy ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-300">{confidence}%</span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Risk Score</span>
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${riskScore < 30 ? 'bg-green-500' : riskScore < 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                style={{ width: `${riskScore}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-300">{riskScore}</span>
          </div>
        </div>
      </div>

      {/* Footer hint */}
      <div className="mt-4 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500 text-center">
          Adjust inputs to see model output change
        </p>
      </div>
      </div>
    </div>
  );
}

const sdkExample = `import { PolicyProofs } from '@arc/policy-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://prover.arc.network'
});

// Agent generates proof before every purchase
const result = await client.prove({
  priceUsdc: 0.05,        // What the agent wants to spend
  budgetUsdc: 1.00,       // Remaining budget
  spentTodayUsdc: 0.20,   // Today's spending
  dailyLimitUsdc: 0.50,   // Policy limit
  serviceSuccessRate: 0.95,
  serviceTotalCalls: 100,
  purchasesInCategory: 5,
  timeSinceLastPurchase: 2.5,
});

// Proof attests: "This agent followed its spending policy"
console.log(result.decision.shouldBuy); // true
console.log(result.proofHash);          // 0x7a8b...`;

const cliExample = `# Generate a proof for agent spending decision
$ arc-prove prove \\
    --price 0.05 \\
    --budget 1.0 \\
    --spent 0.2 \\
    --limit 0.5

Decision: APPROVE
Confidence: 92%
Proof hash: 0x7a8b3c4d...
Proof size: 48.5 KB
Generated in 6.2s

# Verify proof attestation on Arc
$ arc-prove check-attestation 0x7a8b3c4d...
✓ Proof is attested on-chain`;

const verifyExample = `// Merchant/protocol verifies agent followed policy
const verification = await client.verify(proof, claimedInputs);

if (!verification.valid) {
  // Agent tried to modify inputs after proof generation
  console.error('Policy violation detected');
  console.error('Expected:', verification.expectedInputHash);
  console.error('Actual:', verification.actualInputHash);
  rejectTransaction();
}

// Check attestation on Arc chain
import { isProofAttested } from '@arc/policy-proofs';
const attested = await isProofAttested(proof.proofHash);`;

export default function Home() {
  const [activeTab, setActiveTab] = useState<'sdk' | 'cli'>('sdk');

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5" />
              </div>
              <span className="font-semibold text-lg">Arc Spending Policy Proofs</span>
            </Link>
            <div className="hidden md:flex items-center gap-6 text-sm text-gray-400">
              <Link href="#problem" className="hover:text-white transition-colors">Problem</Link>
              <Link href="#solution" className="hover:text-white transition-colors">Solution</Link>
              <Link href="#model" className="hover:text-white transition-colors">Model</Link>
              <Link href="#why-arc" className="hover:text-white transition-colors">Why Arc</Link>
              <Link href="#why-jolt" className="hover:text-white transition-colors">Why Jolt-Atlas</Link>
              <Link href="#roadmap" className="hover:text-white transition-colors">Roadmap</Link>
              <Link href="/demo" className="hover:text-white transition-colors">Demo</Link>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/demo"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Try Demo
            </Link>
            <a
              href="#quickstart"
              className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Get Started
            </a>
          </div>
        </div>
      </nav>

      {/* Hero - Agentic Commerce Focus */}
      <section className="pt-24 pb-16 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            {/* Left: Title and CTA */}
            <div>
              <div className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/20 rounded-full px-3 py-1 text-sm text-purple-400 mb-6">
                <Bot className="w-4 h-4" />
                Agentic Commerce Infrastructure
              </div>
              <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
                zkML Spending Proofs for
                <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-purple-600">
                  Agentic Commerce on Arc
                </span>
              </h1>
              <p className="text-xl text-gray-400 mb-8 leading-relaxed">
                Jolt-Atlas zkML spending policy proofs let agents prove they followed spending rules—cryptographically—before
                every USDC transaction on Arc.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 mb-8">
                <a
                  href="#quickstart"
                  className="inline-flex items-center justify-center gap-2 bg-white text-black px-6 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors"
                >
                  Get Started
                  <ArrowRight className="w-4 h-4" />
                </a>
                <Link
                  href="/demo"
                  className="inline-flex items-center justify-center gap-2 border border-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-white/5 transition-colors"
                >
                  View Demo
                </Link>
              </div>
              {/* Install command */}
              <div className="max-w-md">
                <InstallCommand pkg="@arc/policy-proofs" />
              </div>
            </div>

            {/* Right: Interactive Model Explorer */}
            <div className="flex justify-center lg:justify-center">
              <ModelExplorer />
            </div>
          </div>
        </div>
      </section>

      {/* Problem Statement */}
      <section id="problem" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-3xl mb-12">
            <h2 className="text-3xl font-bold mb-4">The Agentic Commerce Problem</h2>
            <p className="text-gray-400 text-lg">
              As AI agents gain economic autonomy, a critical question emerges:
              <span className="text-white font-medium"> How do you trust an agent with your money?</span>
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-6">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center mb-4">
                <Bot className="w-5 h-5 text-red-400" />
              </div>
              <h3 className="font-semibold mb-2 text-red-300">Unverifiable Decisions</h3>
              <p className="text-sm text-gray-400">
                Agents make spending decisions in black boxes. Users can&apos;t verify
                the agent actually followed its policy before spending.
              </p>
            </div>

            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-6">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center mb-4">
                <DollarSign className="w-5 h-5 text-red-400" />
              </div>
              <h3 className="font-semibold mb-2 text-red-300">Unpredictable Costs</h3>
              <p className="text-sm text-gray-400">
                Traditional chains have volatile gas fees. Agents need
                predictable transaction costs to make autonomous financial decisions.
              </p>
            </div>

            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-6">
              <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center mb-4">
                <Globe className="w-5 h-5 text-red-400" />
              </div>
              <h3 className="font-semibold mb-2 text-red-300">Machine-Hostile Rails</h3>
              <p className="text-sm text-gray-400">
                Current financial infrastructure was built for humans.
                Agents need fast, cheap, and programmable stablecoin rails.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Solution */}
      <section id="solution" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            <div>
              <h2 className="text-3xl font-bold mb-4">Cryptographic Policy Compliance</h2>
              <p className="text-gray-400 mb-6">
                Every agent spending decision generates a SNARK proof. This proof mathematically
                guarantees the agent&apos;s ML model evaluated the purchase against its policy—without
                revealing the policy logic itself.
              </p>

              {/* Tabs */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setActiveTab('sdk')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === 'sdk'
                      ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <Code className="w-4 h-4" />
                  SDK
                </button>
                <button
                  onClick={() => setActiveTab('cli')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === 'cli'
                      ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <Terminal className="w-4 h-4" />
                  CLI
                </button>
              </div>

              <CodeBlock code={activeTab === 'sdk' ? sdkExample : cliExample} />
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-4">What the Proof Guarantees</h3>
              <div className="space-y-4">
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Check className="w-4 h-4 text-green-400" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Policy Was Evaluated</h4>
                      <p className="text-sm text-gray-400">
                        The spending model ran on the claimed inputs. No shortcuts, no bypasses.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Check className="w-4 h-4 text-green-400" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Decision Matches Output</h4>
                      <p className="text-sm text-gray-400">
                        The approve/reject decision came from the model, not fabricated after.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Check className="w-4 h-4 text-green-400" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Inputs Are Locked</h4>
                      <p className="text-sm text-gray-400">
                        Hash of inputs baked into proof. Any tampering is cryptographically detectable.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-purple-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Lock className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">Logic Stays Private</h4>
                      <p className="text-sm text-gray-400">
                        Proof reveals nothing about model weights or decision thresholds.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Spending Policy Model */}
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

      {/* Why Arc */}
      <section id="why-arc" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Arc for Agentic Commerce</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Arc is purpose-built for the machine economy. Every design decision
              optimizes for autonomous agents operating with stablecoins.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4">
                <DollarSign className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="font-semibold mb-2">USDC Native Gas</h3>
              <p className="text-sm text-gray-400">
                Pay fees in stablecoins. Agents don&apos;t need volatile tokens—predictable
                costs for every transaction.
              </p>
            </div>

            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="font-semibold mb-2">Sub-Second Finality</h3>
              <p className="text-sm text-gray-400">
                Deterministic confirmation times. Agents can chain transactions
                without waiting for block confirmations.
              </p>
            </div>

            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center mb-4">
                <Lock className="w-5 h-5 text-purple-400" />
              </div>
              <h3 className="font-semibold mb-2">Optional Privacy</h3>
              <p className="text-sm text-gray-400">
                Confidential transactions when needed. Agents can protect
                sensitive business logic and trading strategies.
              </p>
            </div>

            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="w-10 h-10 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
                <Globe className="w-5 h-5 text-amber-400" />
              </div>
              <h3 className="font-semibold mb-2">Enterprise Infrastructure</h3>
              <p className="text-sm text-gray-400">
                Circle-backed with StableFX and Payments Network.
                Production-grade rails for real-world commerce.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Why Jolt-Atlas zkML */}
      <section id="why-jolt" className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Jolt-Atlas zkML</h2>
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
              <h3 className="font-semibold mb-2">3-7x Faster</h3>
              <p className="text-sm text-gray-400">
                Benchmarks show ~0.7s proving time vs 2-5+ seconds for competing zkML frameworks.
                Sub-second proofs for real-time agent decisions.
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
                Production-ready polynomial commitment scheme. 143ms verification time
                with compact proof sizes suitable for on-chain attestation.
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

          <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6 max-w-2xl mx-auto">
            <h4 className="font-semibold mb-4 text-center">Spending Model Performance</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Proof Generation</span>
                <span className="font-mono text-cyan-400">~0.7s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Verification</span>
                <span className="font-mono text-cyan-400">143ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Model Inputs</span>
                <span className="font-mono text-white">8 features</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Model Outputs</span>
                <span className="font-mono text-white">3 values</span>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700 text-center">
              <a
                href="https://github.com/ICME-Lab/jolt-atlas"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300"
              >
                View JOLT Atlas on GitHub
                <ArrowRight className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Roadmap */}
      <section id="roadmap" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Roadmap</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Building toward full-stack zkML for autonomous agents.
              Each milestone unlocks new capabilities for agentic commerce.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Now */}
            <div className="bg-[#0a0a0a] border border-green-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-green-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                NOW
              </div>
              <div className="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="font-semibold mb-2">Single-Agent Proofs</h3>
              <p className="text-sm text-gray-400 mb-4">
                Production-ready zkML for individual agent spending decisions with sub-second proof generation.
              </p>
              <ul className="space-y-1 text-xs text-gray-500">
                <li className="flex items-center gap-1.5">
                  <Check className="w-3 h-3 text-green-400" />
                  MLP policy models
                </li>
                <li className="flex items-center gap-1.5">
                  <Check className="w-3 h-3 text-green-400" />
                  8-feature input vectors
                </li>
                <li className="flex items-center gap-1.5">
                  <Check className="w-3 h-3 text-green-400" />
                  On-chain attestation
                </li>
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
              <h3 className="font-semibold mb-2">On-Chain Verification</h3>
              <p className="text-sm text-gray-400 mb-4">
                Full cryptographic SNARK verification on Arc—not just attestation, but mathematical proof validation.
              </p>
              <ul className="space-y-1 text-xs text-gray-500">
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-purple-400" />
                  HyperKZG verifier contract
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-purple-400" />
                  BN254 pairing precompile
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-purple-400" />
                  Trustless policy enforcement
                </li>
              </ul>
            </div>

            {/* Future */}
            <div className="bg-[#0a0a0a] border border-cyan-500/30 rounded-xl p-6 relative">
              <div className="absolute -top-3 left-4 bg-cyan-500 text-black text-xs font-semibold px-2 py-0.5 rounded">
                FUTURE
              </div>
              <div className="w-10 h-10 bg-cyan-500/10 rounded-lg flex items-center justify-center mb-4 mt-2">
                <Layers className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold mb-2">Batched &amp; Advanced Models</h3>
              <p className="text-sm text-gray-400 mb-4">
                Multi-sample batching and expanded model support for high-throughput agent operations.
              </p>
              <ul className="space-y-1 text-xs text-gray-500">
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-cyan-400" />
                  Batched inference
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-cyan-400" />
                  Attention layers
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-cyan-400" />
                  F16 quantization
                </li>
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
              <h3 className="font-semibold mb-2">Agent Swarm Proofs</h3>
              <p className="text-sm text-gray-400 mb-4">
                Coordinated multi-agent spending with collective policy enforcement and shared trust.
              </p>
              <ul className="space-y-1 text-xs text-gray-500">
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-amber-400" />
                  Multi-agent coordination
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-amber-400" />
                  Recursive proofs
                </li>
                <li className="flex items-center gap-1.5">
                  <ArrowRight className="w-3 h-3 text-amber-400" />
                  Cross-chain attestation
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Architecture */}
      <section className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            <div>
              <h2 className="text-3xl font-bold mb-4">Verify Before You Trust</h2>
              <p className="text-gray-400 mb-6">
                Merchants and protocols can verify agent policy compliance before
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

      {/* Quickstart */}
      <section id="quickstart" className="py-16 px-6 border-t border-gray-800 bg-[#0d1117]/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Start Building</h2>
            <p className="text-gray-400">Add policy proofs to your agent in minutes</p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
                  <Code className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h3 className="font-semibold">TypeScript SDK</h3>
                  <p className="text-sm text-gray-400">For agent frameworks</p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg px-4 py-2 font-mono text-sm">
                  <span className="text-gray-500">$ </span>
                  <span className="text-gray-300">npm install @arc/policy-proofs</span>
                </div>
                <a
                  href="https://github.com"
                  className="inline-flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
                >
                  View documentation
                  <ArrowRight className="w-3 h-3" />
                </a>
              </div>
            </div>

            <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
                  <Terminal className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h3 className="font-semibold">CLI Tool</h3>
                  <p className="text-sm text-gray-400">For testing and CI/CD</p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="bg-[#0d1117] border border-gray-800 rounded-lg px-4 py-2 font-mono text-sm">
                  <span className="text-gray-500">$ </span>
                  <span className="text-gray-300">npm install -g @arc/policy-proofs-cli</span>
                </div>
                <a
                  href="https://github.com"
                  className="inline-flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
                >
                  View documentation
                  <ArrowRight className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo CTA */}
      <section className="py-16 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">See It In Action</h2>
          <p className="text-gray-400 mb-8 max-w-xl mx-auto">
            Generate real SNARK proofs, test tamper detection, and run
            end-to-end agent payments on Arc Testnet.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/demo/playground"
              className="inline-flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 px-6 py-3 rounded-lg font-medium transition-colors"
            >
              Try the Playground
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/demo/tamper"
              className="inline-flex items-center justify-center gap-2 border border-gray-700 px-6 py-3 rounded-lg font-medium hover:bg-white/5 transition-colors"
            >
              Test Tamper Detection
            </Link>
          </div>
        </div>
      </section>

    </div>
  );
}
