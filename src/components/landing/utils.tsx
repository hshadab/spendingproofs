'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export function CopyButton({ text }: { text: string }) {
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

export function CodeBlock({ code }: { code: string }) {
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

export function InstallCommand({ pkg }: { pkg: string }) {
  return (
    <div className="flex items-center gap-3 bg-[#0d1117] border border-gray-800 rounded-lg px-4 py-3 font-mono text-sm">
      <span className="text-gray-500">$</span>
      <span className="text-gray-300 flex-1">npm install {pkg}</span>
      <CopyButton text={`npm install ${pkg}`} />
    </div>
  );
}

// SDK code example
export const sdkExample = `import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://prover.spendingproofs.dev'
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

// Verify example - verification GATES the transfer
export const verifyExample = `// 1. VERIFY - This gates the transfer (trust boundary)
const verification = await client.verify(
  proof.proof,
  proof.programIo,  // Required for cryptographic verification
  'spending-model'
);

if (!verification.valid) {
  throw new Error('Proof verification failed - blocking transfer');
}

// 2. ATTEST - Record on-chain for transparency (audit trail)
import { submitAttestation } from '@hshadab/spending-proofs';
await submitAttestation(proof.proofHash);

// 3. TRANSFER - Execute via SpendingGateWallet
import { useSpendingGateWallet } from './hooks';
const { gatedTransfer } = useSpendingGateWallet(walletAddress);

await gatedTransfer({
  to: merchantAddress,
  amount: parseUSDC('0.05'),
  proofHash: proof.proofHash,
  expiry: BigInt(Date.now() / 1000 + 3600),
});`;
