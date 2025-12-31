'use client';

import { Bot, DollarSign, Globe, Check, Lock } from 'lucide-react';

export function ProblemSolutionSection() {
  return (
    <section id="problem" className="py-16 px-6 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Left: Problem */}
          <div>
            <h2 className="text-2xl font-bold mb-3">The Agentic Commerce Problem</h2>
            <p className="text-gray-400 mb-6">
              As AI agents gain economic autonomy, a critical question emerges:
              <span className="text-white font-medium"> How do you trust an agent with your money?</span>
            </p>

            <div className="space-y-4">
              <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-red-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-red-300 text-sm">Unverifiable Decisions</h3>
                    <p className="text-xs text-gray-400 mt-1">
                      Agents make spending decisions in black boxes. Users can&apos;t verify
                      the agent actually followed its policy.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <DollarSign className="w-4 h-4 text-red-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-red-300 text-sm">Unpredictable Costs</h3>
                    <p className="text-xs text-gray-400 mt-1">
                      Traditional chains have volatile gas fees. Agents need
                      predictable transaction costs.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-red-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Globe className="w-4 h-4 text-red-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-red-300 text-sm">Machine-Hostile Rails</h3>
                    <p className="text-xs text-gray-400 mt-1">
                      Current financial infrastructure was built for humans.
                      Agents need programmable stablecoin rails.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Solution */}
          <div id="solution">
            <h2 className="text-2xl font-bold mb-3">The Solution: Cryptographic Policy Compliance</h2>
            <p className="text-gray-400 mb-6">
              Every spending decision generates a SNARK proof—mathematically guaranteeing
              policy compliance without revealing the policy logic.
            </p>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 bg-green-500/10 rounded flex items-center justify-center flex-shrink-0">
                    <Check className="w-3 h-3 text-green-400" />
                  </div>
                  <h4 className="font-medium text-sm">Policy Evaluated</h4>
                </div>
                <p className="text-xs text-gray-400">
                  Model ran on claimed inputs. No shortcuts.
                </p>
              </div>

              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 bg-green-500/10 rounded flex items-center justify-center flex-shrink-0">
                    <Check className="w-3 h-3 text-green-400" />
                  </div>
                  <h4 className="font-medium text-sm">Output Verified</h4>
                </div>
                <p className="text-xs text-gray-400">
                  Decision came from the model, not fabricated.
                </p>
              </div>

              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 bg-green-500/10 rounded flex items-center justify-center flex-shrink-0">
                    <Check className="w-3 h-3 text-green-400" />
                  </div>
                  <h4 className="font-medium text-sm">Inputs Locked</h4>
                </div>
                <p className="text-xs text-gray-400">
                  Hash baked into proof. Tampering detected.
                </p>
              </div>

              <div className="bg-[#0d1117] border border-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 bg-purple-500/10 rounded flex items-center justify-center flex-shrink-0">
                    <Lock className="w-3 h-3 text-purple-400" />
                  </div>
                  <h4 className="font-medium text-sm">Logic Private</h4>
                </div>
                <p className="text-xs text-gray-400">
                  Model weights and thresholds stay hidden.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
