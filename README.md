# Spending Proofs

**zkML Spending Proofs for Autonomous Agents on Arc**

Cryptographically prove your agent evaluated its spending policy—without revealing the policy itself. Built on Jolt-Atlas. Designed for Arc.

## Architecture

Three components work together:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│      YOUR APP       │     │       PROVER        │     │     ARC CHAIN       │
│   (SDK installed)   │────▶│   (Jolt-Atlas)      │────▶│   (Settlement)      │
│                     │     │                     │     │                     │
│ npm install         │     │ • Runs neural net   │     │ • Proof attestation │
│ @arc/policy-proofs  │     │ • Generates SNARK   │     │ • USDC transfer     │
│                     │     │ • ~2s proving time  │     │ • Sub-sec finality  │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Quick Start

```bash
npm install @arc/policy-proofs
```

```typescript
import { PolicyProofs } from '@arc/policy-proofs';

const client = new PolicyProofs({
  proverUrl: 'https://prover.spendingproofs.dev'
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

## Hosting Options

### SDK (Your App)
- npm package, works anywhere (Node.js, browser, React)
- Zero infrastructure required from you

### Prover Service

**Option 1: Hosted (Recommended)**
```typescript
const client = new PolicyProofs({
  proverUrl: 'https://prover.spendingproofs.dev'
});
```
- Free tier: 100 proofs/day
- No setup required
- Always up-to-date models

**Option 2: Self-Hosted**
```bash
docker run -p 3001:3001 ghcr.io/arc/spending-prover
```
- Full data privacy
- No rate limits
- Requirements: 8GB RAM, 4 cores

## Spending Model

The policy model evaluates 8 inputs and produces 3 outputs:

### Inputs

| Field | Type | Description |
|-------|------|-------------|
| `priceUsdc` | number | Purchase amount in USDC |
| `budgetUsdc` | number | Remaining agent budget |
| `spentTodayUsdc` | number | Today's cumulative spending |
| `dailyLimitUsdc` | number | Policy-defined daily limit |
| `serviceSuccessRate` | number | Historical success rate (0-1) |
| `serviceTotalCalls` | number | Total calls to this service |
| `purchasesInCategory` | number | Category purchase count |
| `timeSinceLastPurchase` | number | Hours since last purchase |

### Outputs

| Field | Type | Description |
|-------|------|-------------|
| `shouldBuy` | boolean | Binary approval decision |
| `confidence` | number | Decision confidence (0-100%) |
| `riskScore` | number | Risk assessment (0-100) |

## Arc Testnet

Contracts are already deployed:

| Contract | Address |
|----------|---------|
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| ArcAgent (Demo) | `0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384` |

**Network Details:**
- Chain ID: `5042002`
- RPC: `https://rpc.testnet.arc.network`
- Explorer: `https://testnet.arcscan.app`

## Why Arc?

| Requirement | Why It's Necessary |
|-------------|-------------------|
| **USDC Gas** | Agents budget in stablecoins. Volatile gas breaks autonomous operation. |
| **Deterministic Finality** | Sub-second finality enables real-time agent commerce. No reorgs. |
| **Opt-in Privacy** | Protect agent strategies from front-running and MEV. |
| **Enterprise Rails** | Circle-backed infrastructure for production deployment. |

## Roadmap

| Phase | Status | Gas Cost | Description |
|-------|--------|----------|-------------|
| **Proof Attestation** | ✅ Now | ~21k | Off-chain proof, on-chain hash |
| **Solidity Verifier** | Next | ~500k | Full on-chain HyperKZG verification |
| **Native Precompile** | Future | ~50k | Arc-native Jolt verifier (10x cheaper) |

## Local Development

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

## Project Structure

```
src/
├── app/
│   ├── page.tsx                    # Landing page with tabs
│   ├── api/
│   │   ├── prove/                  # Proof generation API
│   │   └── demo/transaction/       # Demo wallet transactions
│   └── demo/
│       ├── playground/             # Interactive proof generation
│       ├── payment/                # End-to-end payment flow
│       └── tamper/                 # Tamper detection demo
├── components/
│   ├── landing/                    # Landing page sections
│   └── EnforcementDemo.tsx         # SpendingGate demo
├── hooks/
│   ├── useProofGeneration.ts       # Proof generation hook
│   └── useDemoWallet.ts            # Demo wallet hook
├── lib/
│   ├── spendingModel.ts            # Spending policy model
│   ├── wagmi.ts                    # Arc chain config
│   └── contracts.ts                # Contract addresses/ABIs
└── providers/
    └── WalletProvider.tsx          # RainbowKit + wagmi

sdk/
├── src/
│   ├── client.ts                   # PolicyProofs client
│   ├── wallet.ts                   # SpendingProofsWallet
│   ├── react.ts                    # React hooks
│   └── wagmi.ts                    # Wagmi integration
└── README.md                       # SDK documentation

docs/
├── sdk-reference.md                # Full API reference
├── arc-integration.md              # Arc integration guide
└── precompile-spec.md              # Native precompile spec
```

## Documentation

- [SDK Reference](./docs/sdk-reference.md) - TypeScript SDK documentation
- [Arc Integration Guide](./docs/arc-integration.md) - Integration patterns and examples
- [Precompile Specification](./docs/precompile-spec.md) - Future native precompile spec

## How It Works

```
1. PROVE     SDK calls prover with 8 spending inputs
2. GENERATE  Prover returns decision + 48KB SNARK proof
3. SETTLE    Submit proof hash + execute USDC payment on Arc
```

## Deployment

This project is designed for server-side deployment (requires API routes):

```bash
# Build
npm run build

# Start production server
npm start
```

Compatible with: Render, Vercel, Railway, or any Node.js host.

## Environment Variables

```env
# Prover
NEXT_PUBLIC_PROVER_URL=https://prover.spendingproofs.dev

# Arc Testnet
NEXT_PUBLIC_ARC_RPC=https://rpc.testnet.arc.network
NEXT_PUBLIC_PROOF_ATTESTATION=0xBE9a5DF7C551324CB872584C6E5bF56799787952

# Demo Wallet (optional - for frictionless demos)
DEMO_WALLET_PRIVATE_KEY=0x...
NEXT_PUBLIC_DEMO_WALLET_ADDRESS=0x...

# USDC on Arc Testnet
NEXT_PUBLIC_USDC_ADDRESS=0x...
```

## License

MIT
