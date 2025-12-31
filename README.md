# Spending Proofs

![Status](https://img.shields.io/badge/status-testnet%20alpha-cyan) ![Arc](https://img.shields.io/badge/built%20for-Arc-purple) ![License](https://img.shields.io/badge/license-MIT-green)

**zkML Spending Proofs for Autonomous Agents on Arc**

Cryptographically prove your agent evaluated its spending policy—without revealing the policy itself. Built on Jolt-Atlas. Designed for Arc.

> **Testnet Alpha** — Core infrastructure is live on Arc testnet. Real zkML proofs, real contracts. Ready for developer preview and feedback.

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
│ @icme-labs/         │     │ • Runs neural net   │     │ • Proof attestation │
│ spending-proofs     │     │ • Generates SNARK   │     │ • USDC transfer     │
│                     │     │ • 4-12s proving     │     │ • Sub-sec finality  │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## What's Real

| Component | Status | Notes |
|-----------|--------|-------|
| Jolt-Atlas SNARK Prover | **Live** | Rust, ~48KB proofs, hosted on Render |
| SDK (`@hshadab/spending-proofs`) | **Live** | TypeScript, React hooks, wagmi |
| Arc Testnet Contracts | **Deployed** | ProofAttestation at `0xBE9a...7952` |
| Circle Programmable Wallets | **Integrated** | API routes ready |
| SpendingGate (Enforcement) | Mock | TypeScript simulation for demo |

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

# Run tests
npm test

# Build for production
npm run build
```

## Arc Testnet

| Contract | Address |
|----------|---------|
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| ArcAgent (Demo) | `0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384` |

- **Chain ID:** `5042002`
- **RPC:** `https://rpc.testnet.arc.network`
- **Explorer:** `https://testnet.arcscan.app`

## License

MIT
