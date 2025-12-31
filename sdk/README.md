# @hshadab/spending-proofs

![Status](https://img.shields.io/badge/status-testnet%20alpha-cyan) ![Arc](https://img.shields.io/badge/Arc%20primitive-purple)

**A spending guardrail primitive for Arc's agent economy.** Generate zkML proofs that your agent followed its spending policy—enabling trustless machine-to-machine payments.

> **Testnet Alpha** — Real zkML proofs on Arc testnet. Ready for developer preview.

## Installation

```bash
npm install @hshadab/spending-proofs
```

## Quick Start

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://spendingproofs-prover.onrender.com'
});

// Generate a proof
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

console.log(result.decision.shouldBuy); // true
console.log(result.proofHash); // 0x...
```

## Features

- **SNARK Proofs**: Generate real zkML proofs using JOLT-Atlas (HyperKZG/BN254)
- **On-Chain Verification**: Verify proofs on Arc Testnet
- **Local Decisions**: Fast local policy evaluation without proof overhead
- **Wallet Integration**: High-level wallet SDK for proof-gated transfers
- **React Hooks**: Ready-to-use hooks for React applications
- **Wagmi Support**: Native integration with wagmi for web3 apps
- **TypeScript Native**: Full type safety and IntelliSense

## Wallet SDK

High-level wallet class for proof-gated spending:

```typescript
import { SpendingProofsWallet } from '@hshadab/spending-proofs/wallet';

const wallet = new SpendingProofsWallet({
  proverUrl: 'https://prover.spendingproofs.dev',
  agentAddress: '0x...',
  chainId: 5042002, // Arc Testnet
});

// Prepare a gated transfer
const result = await wallet.prepareGatedTransfer({
  recipient: '0x...',
  amountUsdc: 50,
  input: {
    priceUsdc: 50,
    budgetUsdc: 100,
    spentTodayUsdc: 20,
    dailyLimitUsdc: 200,
    serviceSuccessRate: 0.95,
    serviceTotalCalls: 100,
    purchasesInCategory: 3,
    timeSinceLastPurchase: 1.5,
  },
});

if (result.approved) {
  console.log('Proof hash:', result.proof.proofHash);
  console.log('TX intent hash:', result.txIntentHash);
  // Execute the transfer with your wallet...
}
```

## React Integration

```tsx
import { useState, useCallback, useMemo } from 'react';
import { PolicyProofs, SpendingInput, ProofResult } from '@hshadab/spending-proofs';

function useSpendingProofs(proverUrl: string) {
  const client = useMemo(() => new PolicyProofs({ proverUrl }), [proverUrl]);
  const [proof, setProof] = useState<ProofResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const generateProof = useCallback(async (input: SpendingInput) => {
    setIsLoading(true);
    try {
      const result = await client.prove(input);
      setProof(result);
      return result;
    } catch (e) {
      setError(e instanceof Error ? e : new Error('Failed'));
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, [client]);

  return { proof, isLoading, error, generateProof };
}
```

## Wagmi Integration

```typescript
import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import {
  getGatedTransferArgs,
  getExplorerTxUrl,
  parseUSDC
} from '@hshadab/spending-proofs/wagmi';
import { SpendingProofsWallet } from '@hshadab/spending-proofs/wallet';

// Prepare proof and execute gated transfer
const wallet = new SpendingProofsWallet({ proverUrl, agentAddress });
const { writeContract, data: hash } = useWriteContract();

async function executeGatedTransfer() {
  const result = await wallet.prepareGatedTransfer({ ... });

  if (result.approved) {
    writeContract(getGatedTransferArgs({
      spendingGateAddress: '0x...',
      recipient: '0x...',
      amountWei: parseUSDC(50),
      proofHash: result.proof.proofHash as `0x${string}`,
      txIntentHash: result.txIntentHash as `0x${string}`,
      nonce: result.txIntent.nonce,
      expiry: BigInt(result.txIntent.expiry),
    }));
  }
}
```

## API Reference

### PolicyProofs

Main client for generating and verifying proofs.

```typescript
const client = new PolicyProofs({
  proverUrl: string;      // Prover service URL
  timeout?: number;       // Request timeout (default: 120000ms)
  fetch?: typeof fetch;   // Custom fetch implementation
});
```

#### prove(input, tag?)

Generate a SNARK proof for a spending decision.

```typescript
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

// result: {
//   proof: string;           // Hex-encoded SNARK proof
//   proofHash: string;       // Proof hash for on-chain verification
//   decision: {
//     shouldBuy: boolean;    // Approve/reject decision
//     confidence: number;    // Confidence level (0-1)
//     riskScore: number;     // Risk score (0-1)
//   };
//   metadata: {
//     modelHash: string;
//     inputHash: string;
//     outputHash: string;
//     proofSize: number;
//     generationTimeMs: number;
//   };
// }
```

#### verify(proof, inputs)

Verify a proof against expected inputs (tamper detection).

```typescript
const verification = await client.verify(proof, originalInputs);
// { valid: boolean, message: string }
```

#### decide(input)

Run spending decision locally without generating a proof.

```typescript
const decision = client.decide(input);
// { shouldBuy: boolean, confidence: number, riskScore: number }
```

### On-Chain Verification

```typescript
import { isProofAttested, ARC_TESTNET } from '@hshadab/spending-proofs';

// Check if proof is attested on Arc Testnet
const isValid = await isProofAttested(proofHash);
```

## Spending Model Inputs

| Input | Type | Description |
|-------|------|-------------|
| priceUsdc | number | Purchase price in USDC |
| budgetUsdc | number | Available budget in USDC |
| spentTodayUsdc | number | Amount spent today in USDC |
| dailyLimitUsdc | number | Daily spending limit in USDC |
| serviceSuccessRate | number | Service reliability (0-1) |
| serviceTotalCalls | number | Total calls to service |
| purchasesInCategory | number | Purchases in this category |
| timeSinceLastPurchase | number | Hours since last purchase |

## Arc Testnet

```typescript
import { ARC_TESTNET } from '@hshadab/spending-proofs';

// ARC_TESTNET = {
//   chainId: 5042002,
//   name: 'Arc Testnet',
//   rpcUrl: 'https://rpc.testnet.arc.network',
//   explorerUrl: 'https://testnet.arcscan.app',
//   contracts: {
//     proofAttestation: '0xBE9a5DF7C551324CB872584C6E5bF56799787952',
//     spendingGateWallet: '0x6A47D13593c00359a1c5Fc6f9716926aF184d138',
//     mockUsdc: '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213',
//   },
// }
```

## Why Arc?

Spending proofs require Arc's unique infrastructure:

| Feature | Arc | Other L2s |
|---------|-----|-----------|
| Gas Token | USDC (stable) | ETH (volatile) |
| Finality | <1s deterministic | ~7 days soft |
| Reorg Risk | None | Sequencer-dependent |
| Privacy | Opt-in available | None |

Learn more: [Arc Documentation](https://arc.builders)

## Security

- **inputsHash**: Prevents input tampering after proof generation
- **txIntentHash**: Binds proof to specific transaction intent
- **Nonce**: Prevents proof replay attacks
- **Expiry**: Limits proof validity window
- **PolicyRegistry**: On-chain model hash verification

See our [Security Model](/security) for comprehensive threat analysis.

## License

MIT
