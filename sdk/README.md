# @arc/policy-proofs

zkML spending policy proofs for Arc chain. Generate and verify SNARK proofs for autonomous agent spending decisions.

## Installation

```bash
npm install @arc/policy-proofs
```

## Quick Start

```typescript
import { PolicyProofs } from '@arc/policy-proofs';

const client = new PolicyProofs({
  proverUrl: 'http://localhost:3001'
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
- **TypeScript Native**: Full type safety and IntelliSense

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

#### health()

Check prover service health.

```typescript
const status = await client.health();
// { healthy: boolean, models: string[] }
```

### On-Chain Verification

```typescript
import { isProofValidOnChain, ARC_TESTNET } from '@arc/policy-proofs';

// Check if proof is valid on Arc Testnet
const isValid = await isProofValidOnChain(proofHash);

// With custom options
const isValid = await isProofValidOnChain(proofHash, {
  contractAddress: '0x...',
  rpcUrl: 'https://rpc.testnet.arc.network',
});
```

### Utilities

```typescript
import {
  spendingInputToArray,    // Convert input object to array
  arrayToSpendingInput,    // Convert array to input object
  validateSpendingInput,   // Validate input values
  formatProofHash,         // Format hash for display
  getExplorerTxUrl,        // Get Arc explorer URL for tx
  defaultPolicy,           // Get default spending policy
} from '@arc/policy-proofs';
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
import { ARC_TESTNET } from '@arc/policy-proofs';

// ARC_TESTNET = {
//   chainId: 5042002,
//   name: 'Arc Testnet',
//   rpcUrl: 'https://rpc.testnet.arc.network',
//   explorerUrl: 'https://testnet.arcscan.app',
//   contracts: {
//     proofAttestation: '0xBE9a5DF7C551324CB872584C6E5bF56799787952',
//   },
// }
```

## With Viem/Wagmi

```typescript
import { getProofAttestationContract } from '@arc/policy-proofs';
import { createPublicClient, http } from 'viem';

const client = createPublicClient({
  chain: arcTestnet,
  transport: http(),
});

const contract = getProofAttestationContract();

const isValid = await client.readContract({
  ...contract,
  functionName: 'isProofValid',
  args: [proofHash as `0x${string}`],
});
```

## License

MIT
