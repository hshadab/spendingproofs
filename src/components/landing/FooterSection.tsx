'use client';

import { Shield } from 'lucide-react';

export function FooterSection() {
  return (
    <footer className="border-t border-gray-800 py-8 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-purple-500 to-purple-700 rounded flex items-center justify-center">
              <Shield className="w-4 h-4" />
            </div>
            <span className="text-sm text-gray-400">Spending Policy Proofs</span>
          </div>
          <p className="text-xs text-gray-500 text-center">
            Cryptographic spending guardrails for autonomous agents
          </p>
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <a
              href="https://www.novanet.xyz/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              NovaNet
            </a>
            <a
              href="https://github.com/ICME-Lab/jolt-atlas"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              Jolt-Atlas
            </a>
            <a
              href="https://github.com/hshadab/spendingproofs"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
