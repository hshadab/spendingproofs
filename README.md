# Spending Policy Proofs

![Status](https://img.shields.io/badge/status-testnet%20alpha-cyan) ![Base Sepolia](https://img.shields.io/badge/Base%20Sepolia-blue) ![License](https://img.shields.io/badge/license-MIT-green)

**Cryptographic spending guardrails for AI agents**

zkML infrastructure that proves AI agents followed their spending policies — enabling trustless, verifiable agentic commerce.

> **Testnet Alpha** — Live on Base Sepolia with Crossmint wallet integration. Real zkML proofs, real transfers.

## On-Chain Controls + ML Verification

**Wallets like Crossmint** provide robust on-chain spending controls:
- Spending limits, multi-sig, role-based permissions
- Programmable policies enforced at the smart contract level

**Some spending decisions need more than simple rules.** "Should we approve this vendor?" depends on risk profile, past performance, budget utilization, and compliance status — evaluated together:
- Vendor risk scoring models
- Historical performance analysis
- Multi-factor approval matrices

```solidity
// ML inference is infeasible on-chain — gas costs explode, EVM can't run neural networks
require(mlModel.evaluate(vendorRisk, history, budget, compliance) == APPROVE);
```

## The Solution

**zkML extends on-chain controls** to ML-based policies:
- Run complex policy models **off-chain**
- Generate cryptographic proofs of correct execution (~48KB SNARK)
- Verify proofs before authorizing payments

On-chain controls handle rule-based policies. zkML proves ML-based policies were evaluated correctly.

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
  priceUsdc: 4500,
  budgetUsdc: 50000,
  vendorRiskScore: 0.15,
  historicalVendorScore: 0.92,
  categoryBudgetUsdc: 15000,
  categorySpentUsdc: 4200,
  vendorComplianceStatus: true,
});

if (result.decision.shouldBuy) {
  console.log('Proof hash:', result.proofHash);
  // Proof verified → execute payment via Crossmint
}
```

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│      YOUR APP       │     │       PROVER        │     │      CROSSMINT      │
│   (SDK installed)   │────▶│   (JOLT-Atlas)      │────▶│   (MPC Wallets)     │
│                     │     │                     │     │                     │
│ Policy evaluation   │     │ • Runs policy model │     │ • Proof verification│
│ + proof request     │     │ • Generates SNARK   │     │ • Token transfer    │
│                     │     │ • ~48KB, ~10s       │     │ • Audit metadata    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## What's Real

| Component | Status | Notes |
|-----------|--------|-------|
| JOLT-Atlas SNARK Prover | **Live** | Rust, ~48KB proofs |
| SDK (`@hshadab/spending-proofs`) | **Live** | TypeScript, React hooks |
| Base Sepolia Contracts | **Deployed** | ProofAttestation, SpendingGateWallet |
| Crossmint Integration | **Live** | MPC wallets, token transfers |
| Interactive Demo | **Live** | [spendingproofs.com/demo/crossmint](https://spendingproofs.com/demo/crossmint) |

## Performance

| Metric | Value |
|--------|-------|
| Proof generation | 4-12 seconds |
| Proof size | ~48 KB |
| Verification | <150ms |

## Base Sepolia Contracts

| Contract | Address |
|----------|---------|
| ProofAttestation | [`0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213`](https://sepolia.basescan.org/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213) |
| SpendingGateWallet | [`0x6A47D13593c00359a1c5Fc6f9716926aF184d138`](https://sepolia.basescan.org/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138) |
| USDC (Testnet) | [`0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f`](https://sepolia.basescan.org/address/0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f) |

## Arc Testnet Contracts

| Contract | Address |
|----------|---------|
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` |
| TestnetUSDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` |

- **Chain ID:** `5042002`
- **RPC:** `https://rpc.testnet.arc.network`
- **Explorer:** `https://testnet.arcscan.app`

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run contract tests (requires Foundry)
cd contracts && forge test
```

## Demos

### Core Demos

| Demo | Description | Link |
|------|-------------|------|
| **Playground** | Interactive policy configuration and proof generation | [/demo/playground](https://spendingproofs.com/demo/playground) |
| **Payment** | End-to-end payment flow with proof verification | [/demo/payment](https://spendingproofs.com/demo/payment) |
| **Tamper** | Demonstrates tamper detection when inputs are modified | [/demo/tamper](https://spendingproofs.com/demo/tamper) |

### Enterprise Demos

| Demo | Partner | Description | Documentation |
|------|---------|-------------|---------------|
| **Crossmint** | [Crossmint](https://crossmint.com) | Enterprise MPC wallets + zkML policy proofs | [README](./src/app/demo/crossmint/README.md) |
| **Morpho** | [Morpho Blue](https://morpho.org) | DeFi vault management with policy-gated operations | [README](./src/app/demo/morpho/README.md) |
| **OpenMind** | [OpenMind](https://openmind.org) | Autonomous robot payments via x402 protocol | [README](./src/app/demo/openmind/README.md) |
| **ACK** | [Catena Labs](https://catenalabs.com/projects/) | Extends ACK identity + receipts with zkML policy proofs | [README](./src/app/demo/ack/README.md) |

## Documentation

- **[SDK Reference](./docs/sdk-reference.md)** - Full API documentation
- **[Architecture](./docs/ARCHITECTURE.md)** - System design
- **[API Reference](./docs/API.md)** - API endpoints

## Links

- [Crossmint](https://crossmint.com) - Enterprise wallet infrastructure
- [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML proof generation
- [Base Sepolia](https://sepolia.basescan.org) - Testnet explorer

## License

MIT
