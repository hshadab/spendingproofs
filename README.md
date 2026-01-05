# Spending Policy Proofs

![Status](https://img.shields.io/badge/status-testnet%20alpha-cyan) ![Arc](https://img.shields.io/badge/Arc%20primitive-purple) ![License](https://img.shields.io/badge/license-MIT-green)

**Cryptographic spending guardrails for Arc's agent economy**

Spending Policy Proofs is infrastructure for agentic commerce on Arc powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge machine learning (zkML). Agents generate cryptographic proofs that they followed their spending policies—enabling trustless machine-to-machine payments.

> **Testnet Alpha** — Core infrastructure is live on Arc testnet. Real zkML proofs, real contracts. Ready for developer preview.

## Quick Start

```bash
npm install @hshadab/spending-proofs
```

```typescript
import { PolicyProofs } from '@hshadab/spending-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://spendingproofs-prover.onrender.com'
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
  console.log('Proof hash:', result.proofHash);
}
```

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│      YOUR APP       │     │       PROVER        │     │     ARC CHAIN       │
│   (SDK installed)   │────▶│   (Jolt-Atlas)      │────▶│   (Settlement)      │
│                     │     │                     │     │                     │
│ @hshadab/           │     │ • Runs neural net   │     │ • Proof attestation │
│ spending-proofs     │     │ • Generates SNARK   │     │ • USDC transfer     │
│                     │     │ • 4-12s proving     │     │ • Sub-sec finality  │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## What's Real

| Component | Status | Notes |
|-----------|--------|-------|
| Jolt-Atlas SNARK Prover | **Live** | Rust, ~48KB proofs, hosted on Render |
| SDK (`@hshadab/spending-proofs`) | **Live** | TypeScript, React hooks, wagmi |
| Arc Testnet Contracts | **Deployed** | ProofAttestation, SpendingGateWallet, TestnetUSDC |
| SpendingGate (Enforcement) | **Live** | Real on-chain enforcement - no proof, no payment |
| Interactive Demo | **Live** | https://spendingproofs.com/demo |

## Performance

| Metric | Value |
|--------|-------|
| Proof generation (warm) | 4-12 seconds |
| Proof generation (cold) | ~30 seconds first proof |
| Proof size | ~48 KB |
| Verification | <150ms off-chain |
| Gas (attestation) | ~21k |

## Documentation

- **[SDK Reference](./docs/sdk-reference.md)** - Full API documentation
- **[Arc Integration](./docs/arc-integration.md)** - Integration patterns
- **[Precompile Spec](./docs/precompile-spec.md)** - Future Arc-native verifier

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run TypeScript tests
npm test

# Run contract tests (requires Foundry)
cd contracts && forge test

# Build for production
npm run build
```

## Features

### Error Handling

Typed error classes for robust error handling:

```typescript
import { ProverError, ValidationError, parseError, ErrorCode } from '@/lib/errors';

try {
  await generateProof(input);
} catch (err) {
  const error = parseError(err);
  switch (error.code) {
    case ErrorCode.PROVER_UNAVAILABLE:
      // Handle prover down
      break;
    case ErrorCode.INVALID_INPUT:
      // Handle validation error
      break;
  }
}
```

### Retry with Exponential Backoff

Automatic retry for transient failures:

```typescript
import { withRetry, proverRetryOptions } from '@/lib/retry';

const result = await withRetry(
  () => fetchProof(inputs),
  {
    ...proverRetryOptions,
    maxAttempts: 5,
    onRetry: (err, attempt) => console.log(`Retry ${attempt}...`),
  }
);
```

### Input Validation

Comprehensive input validation:

```typescript
import { validateSpendingInput, assertValidSpendingInput } from '@/lib/spendingModel';

// Get validation errors
const { valid, errors } = validateSpendingInput(input);

// Or throw on invalid input
assertValidSpendingInput(input); // throws ValidationError if invalid
```

### Proof Caching

LRU cache for proof results (same inputs = same proof):

```typescript
import { withProofCache, getProofCache } from '@/lib/proofCache';

const result = await withProofCache(inputs, 'spending', async () => {
  return await generateProof(inputs);
});

// Check cache stats
const stats = getProofCache().getStats();
console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);
```

### Metrics & Observability

Built-in metrics collection:

```typescript
import { getMetricsCollector } from '@/lib/metrics';

const metrics = getMetricsCollector();
metrics.recordRequest();
metrics.recordSuccess(generationTimeMs);

// Export for Prometheus
console.log(metrics.toPrometheusFormat());
```

### Signature Authentication

Optional wallet signature authentication for API requests:

```typescript
// Enable in .env
REQUIRE_SIGNATURE_AUTH=true
ALLOWED_PROVER_ADDRESSES=0x123...,0x456...

// Use the signed proof generation hook
import { useSignedProofGeneration } from '@/hooks/useSignedProofGeneration';

const { generateSignedProof } = useSignedProofGeneration();
await generateSignedProof(input); // Auto-signs with connected wallet
```

## Arc Testnet

| Contract | Address |
|----------|---------|
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` |
| TestnetUSDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` |
| ArcAgent (Demo) | `0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384` |

- **Chain ID:** `5042002`
- **RPC:** `https://rpc.testnet.arc.network`
- **Explorer:** `https://testnet.arcscan.app`

## License

MIT
