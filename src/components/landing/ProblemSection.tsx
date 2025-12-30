'use client';

import { Bot, DollarSign, Globe } from 'lucide-react';

export function ProblemSection() {
  return (
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
  );
}
