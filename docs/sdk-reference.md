# SDK Reference

The `@hshadab/spending-proofs` SDK provides TypeScript bindings for generating and verifying spending proofs on Arc.

## Installation

```bash
npm install @hshadab/spending-proofs
```

## Quick Start

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

// Initialize client
const client = new PolicyProofs({
  proverUrl: 'https://spendingproofs-prover.onrender.com'
});

// Generate a spending proof
const proof = await client.prove({
  priceUsdc: 0.05,
  budgetUsdc: 1.00,
  spentTodayUsdc: 0.20,
  dailyLimitUsdc: 0.50,
  serviceSuccessRate: 0.95,
  serviceTotalCalls: 100,
  purchasesInCategory: 5,
  timeSinceLastPurchase: 2,
});

console.log('Decision:', proof.decision.shouldBuy);
console.log('Confidence:', proof.decision.confidence);
console.log('Risk Score:', proof.decision.riskScore);
```

## API Reference

### PolicyProofs

Main client class for interacting with the spending proofs system.

#### Constructor

```typescript
new PolicyProofs(config: PolicyProofsConfig)
```

**Config Options:**

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `rpcUrl` | `string` | Yes | Arc RPC endpoint URL |
| `privateKey` | `string` | No | Private key for signing transactions |
| `policyId` | `string` | No | Custom policy ID (default: 'default-spending-policy') |
| `proverUrl` | `string` | No | Custom prover service URL |

#### prove()

Generate a spending proof for the given inputs.

```typescript
async prove(inputs: SpendingInputs): Promise<SpendingProof>
```

**SpendingInputs:**

| Field | Type | Description |
|-------|------|-------------|
| `priceUsdc` | `number` | Purchase amount in USDC |
| `budgetUsdc` | `number` | Remaining budget in USDC |
| `spentTodayUsdc` | `number` | Amount spent today in USDC |
| `dailyLimitUsdc` | `number` | Daily spending limit in USDC |
| `serviceSuccessRate` | `number` | Service success rate (0-1) |
| `serviceTotalCalls` | `number` | Total calls to service |
| `purchasesInCategory` | `number` | Purchases in this category |
| `timeSinceLastPurchase` | `number` | Hours since last purchase |

**Returns: SpendingProof**

| Field | Type | Description |
|-------|------|-------------|
| `proof` | `Uint8Array` | Serialized Jolt proof (~48KB) |
| `decision` | `SpendingDecision` | Model output |
| `inputsHash` | `string` | Poseidon hash of inputs |
| `policyHash` | `string` | Hash of policy verification key |
| `timestamp` | `number` | Proof generation timestamp |

#### verify()

Verify an existing spending proof.

```typescript
async verify(params: VerifyParams): Promise<boolean>
```

**VerifyParams:**

| Field | Type | Description |
|-------|------|-------------|
| `proof` | `Uint8Array` | Serialized proof bytes |
| `policyHash` | `string` | Expected policy hash |
| `inputsHash` | `string` | Expected inputs hash |
| `txIntentHash` | `string` | Transaction intent hash |

#### attest()

Commit a proof attestation to Arc.

```typescript
async attest(proof: SpendingProof): Promise<AttestationResult>
```

**Returns: AttestationResult**

| Field | Type | Description |
|-------|------|-------------|
| `txHash` | `string` | Transaction hash on Arc |
| `blockNumber` | `number` | Block containing attestation |
| `attestationId` | `string` | Unique attestation identifier |

#### proveAndExecute()

Generate proof and execute spending in one call.

```typescript
async proveAndExecute(params: ExecuteParams): Promise<ExecuteResult>
```

**ExecuteParams:**

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `SpendingInputs` | Spending model inputs |
| `recipient` | `string` | USDC recipient address |
| `amount` | `bigint` | Amount in USDC (6 decimals) |
| `expiry` | `number` | Optional expiry timestamp |

**Returns: ExecuteResult**

| Field | Type | Description |
|-------|------|-------------|
| `txHash` | `string` | Transaction hash |
| `proof` | `SpendingProof` | Generated proof |
| `verified` | `boolean` | Whether proof was verified on-chain |
| `verificationMethod` | `string` | 'native-precompile' \| 'solidity-verifier' \| 'attestation' |

### SpendingDecision

Model output from proof generation.

```typescript
interface SpendingDecision {
  shouldBuy: boolean;      // Binary approval decision
  confidence: number;      // 0-100 percentage
  riskScore: number;       // 0-100 risk assessment
}
```

### Types

#### SpendingInputs

```typescript
interface SpendingInputs {
  priceUsdc: number;
  budgetUsdc: number;
  spentTodayUsdc: number;
  dailyLimitUsdc: number;
  serviceSuccessRate: number;
  serviceTotalCalls: number;
  purchasesInCategory: number;
  timeSinceLastPurchase: number;
}
```

#### SpendingProof

```typescript
interface SpendingProof {
  proof: Uint8Array;
  decision: SpendingDecision;
  inputsHash: string;
  policyHash: string;
  timestamp: number;
  metadata: {
    proverVersion: string;
    proofSize: number;
    generationTimeMs: number;
  };
}
```

## Examples

### Basic Proof Generation

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  rpcUrl: 'https://rpc.arc.network',
});

const proof = await client.prove({
  priceUsdc: 0.10,
  budgetUsdc: 5.00,
  spentTodayUsdc: 1.50,
  dailyLimitUsdc: 2.00,
  serviceSuccessRate: 0.98,
  serviceTotalCalls: 500,
  purchasesInCategory: 10,
  timeSinceLastPurchase: 0.5,
});

if (proof.decision.shouldBuy) {
  console.log('Purchase approved with', proof.decision.confidence, '% confidence');
} else {
  console.log('Purchase rejected. Risk score:', proof.decision.riskScore);
}
```

### End-to-End Payment Flow

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  rpcUrl: 'https://rpc.arc.network',
  privateKey: process.env.AGENT_PRIVATE_KEY,
});

// Execute spending with proof
const result = await client.proveAndExecute({
  inputs: {
    priceUsdc: 0.05,
    budgetUsdc: 1.00,
    spentTodayUsdc: 0.20,
    dailyLimitUsdc: 0.50,
    serviceSuccessRate: 0.95,
    serviceTotalCalls: 100,
    purchasesInCategory: 5,
    timeSinceLastPurchase: 2,
  },
  recipient: '0x742d35Cc6634C0532925a3b844Bc9e7595f1b2d1',
  amount: 50000n, // 0.05 USDC (6 decimals)
});

console.log('Transaction:', result.txHash);
console.log('Verification method:', result.verificationMethod);
```

### Merchant Verification

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  rpcUrl: 'https://rpc.arc.network',
});

// Merchant receives proof from agent
const proofBytes = /* received from agent */;
const policyHash = /* expected policy */;
const inputsHash = /* from proof metadata */;
const txIntentHash = /* computed from transaction */;

// Verify before accepting payment
const isValid = await client.verify({
  proof: proofBytes,
  policyHash,
  inputsHash,
  txIntentHash,
});

if (isValid) {
  console.log('Proof verified - accept payment');
} else {
  console.log('Invalid proof - reject payment');
}
```

### Custom Policy Configuration

```typescript
import { PolicyProofs, PolicyConfig } from '@hshadab/spending-proofs';

// Define custom policy thresholds
const policy: PolicyConfig = {
  maxSinglePurchase: 1.00,
  dailyLimit: 10.00,
  minServiceSuccessRate: 0.90,
  maxRiskScore: 50,
  requiredConfidence: 70,
};

const client = new PolicyProofs({
  rpcUrl: 'https://rpc.arc.network',
  policy,
});

// Proofs will use custom policy thresholds
const proof = await client.prove({
  // ... inputs
});
```

### Batch Proof Generation

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  rpcUrl: 'https://rpc.arc.network',
});

// Generate multiple proofs in parallel
const purchases = [
  { priceUsdc: 0.05, /* ... */ },
  { priceUsdc: 0.10, /* ... */ },
  { priceUsdc: 0.15, /* ... */ },
];

const proofs = await Promise.all(
  purchases.map(inputs => client.prove(inputs))
);

// Filter approved purchases
const approved = proofs.filter(p => p.decision.shouldBuy);
console.log(`${approved.length}/${proofs.length} purchases approved`);
```

## Error Handling

```typescript
import { PolicyProofs, SpendingProofError } from '@hshadab/spending-proofs';

try {
  const proof = await client.prove(inputs);
} catch (error) {
  if (error instanceof SpendingProofError) {
    switch (error.code) {
      case 'PROOF_GENERATION_FAILED':
        console.error('Prover service unavailable');
        break;
      case 'INVALID_INPUTS':
        console.error('Invalid input values:', error.details);
        break;
      case 'NETWORK_ERROR':
        console.error('Network error:', error.message);
        break;
      default:
        console.error('Unknown error:', error);
    }
  }
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARC_RPC_URL` | Default Arc RPC endpoint |
| `PROVER_URL` | Custom prover service URL |
| `PRIVATE_KEY` | Agent private key for signing |
| `POLICY_ID` | Custom policy identifier |

## Changelog

### v0.2.0 (Current)

- Real SNARK proof generation via Jolt-Atlas
- On-chain attestation support
- SpendingGateWallet integration
- React hooks and wagmi support
- TypeScript SDK

### v0.3.0 (Planned)

- Solidity verifier integration
- Batch proof generation
- Custom policy configuration

### v1.0.0 (Planned)

- Native precompile support
- Production-ready with full test coverage
- Multi-chain support via CCTP
