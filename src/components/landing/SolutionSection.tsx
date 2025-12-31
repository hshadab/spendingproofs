'use client';

import { Check, Lock } from 'lucide-react';
import { CodeBlock, sdkExample } from './utils';

export function SolutionSection() {
  return (
    <section id="solution" className="py-16 px-6 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <div>
            <h2 className="text-3xl font-bold mb-4">The Solution: Cryptographic Policy Compliance</h2>
            <p className="text-gray-400 mb-6">
              Every agent spending decision generates a SNARK proof. This proof mathematically
              guarantees the agent&apos;s ML model evaluated the purchase against its policy—without
              revealing the policy logic itself.
            </p>

            <CodeBlock code={sdkExample} />
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
  );
}
