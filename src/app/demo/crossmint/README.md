# Verifiable Agentic Commerce: Crossmint + zkML

Extending **Crossmint's on-chain spending controls** with **zkML proofs for ML-based policies** — enabling cryptographically verifiable AI spending decisions.

## Crossmint's On-Chain Policy Enforcement

Crossmint already provides robust on-chain spending controls:
- **Spending limits** enforced at the smart contract level
- **Multi-sig requirements** for high-value transactions
- **Role-based permissions** with delegated signers
- **Programmable policies** stored on-chain, not in opaque TEEs

These work excellently for rule-based policies like `require(amount <= limit)`.

## Extending to ML-Based Policies

**Some spending decisions need more than simple rules.** "Should we approve this vendor?" depends on risk profile, past performance, budget utilization, and compliance status — evaluated together:
- Vendor risk scoring models
- Historical performance analysis
- Multi-factor approval matrices
- Dynamic compliance evaluation

```solidity
// ML inference is infeasible on-chain — gas costs explode, EVM can't run neural networks
require(mlModel.evaluate(vendorRisk, history, budget, compliance) == APPROVE);
```

**zkML extends Crossmint's verification capabilities** to these ML-based policies:
- Run complex policy models **off-chain**
- Generate cryptographic proofs of correct execution (~48KB SNARK)
- Verify proofs before payment execution
- Crossmint handles the actual wallet + transfer infrastructure

## The Integration

**Crossmint** provides enterprise wallet infrastructure:
- Smart wallets with on-chain permissions
- Token transfer APIs with audit metadata
- Gas abstraction and multi-chain support

**zkML** provides cryptographic proof of ML policy execution:
- Proves the exact model ran on exact inputs
- Privacy-preserving (budget details stay hidden)
- Unforgeable, mathematically verifiable

**Together**: Crossmint's trusted payment infrastructure + zkML's ML verification = enterprise-grade agentic commerce

---

## What This Demo Shows

An enterprise procurement agent evaluates a **$4,500/month DataDog APM** subscription:

| Policy Factor | Value | Description |
|--------------|-------|-------------|
| Vendor Risk Score | 15% | Low risk (threshold: 70%) |
| Historical Performance | 92% | Excellent track record |
| Category Budget | $15,000 | Observability allocation |
| Category Spent | $4,200 | Room remaining |
| Compliance Status | Verified | Passed all checks |
| Relationship | 2 years | Established vendor |

**Result**: APPROVE at 94% confidence — with cryptographic proof that all 6 factors were evaluated.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. POLICY EVALUATION (Off-chain ML Model)                      │
│     Agent evaluates 6 vendor factors + 4 budget constraints     │
│     Output: APPROVE/REJECT with confidence score                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. zkML PROOF GENERATION (JOLT-Atlas)                          │
│     SNARK proof generated (~48KB, ~10s)                         │
│     Proves: Exact policy model ran on exact inputs              │
│     Privacy: Budget limits, vendor scores stay hidden           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CROSSMINT TRANSFER (Enterprise Wallet)                      │
│     Proof verified → Payment authorized                         │
│     MPC wallet executes USDC transfer                           │
│     Proof hash stored in transaction metadata                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. AUDIT ATTESTATION (Base Sepolia)                            │
│     Proof hash recorded on-chain                                │
│     Immutable audit trail for compliance                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Crossmint Integration

This demo uses Crossmint's Wallets API for enterprise-grade payment execution.

### MPC Wallet Creation

```typescript
// Create Fireblocks-backed MPC wallet for the agent
POST /2025-06-09/wallets
{
  "type": "evm-mpc-wallet",
  "linkedUser": "procurement-agent-001"
}
```

**Benefits**:
- No private key exposure — keys distributed across MPC nodes
- Fireblocks security infrastructure
- Idempotent creation — same `linkedUser` returns existing wallet

### Token Transfers with Audit Metadata

```typescript
// Execute transfer with proof hash for audit trail
POST /2025-06-09/wallets/{locator}/tokens/base-sepolia:usdc/transfers
{
  "recipient": "0x...",
  "amount": "4500"
}
```

**Benefits**:
- Gas abstraction — Crossmint handles fees
- Multi-chain support — Base, Polygon, Arbitrum, Optimism
- Transaction metadata — link payments to verified proofs

### Integration Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  YOUR APPLICATION                                                 │
│                                                                   │
│  1. Generate zkML proof (JOLT-Atlas)                             │
│  2. Verify proof cryptographically                                │
│  3. If valid → Call Crossmint Transfer API                       │
│  4. Crossmint executes payment                                    │
│  5. Record proof hash for audit                                   │
└──────────────────────────────────────────────────────────────────┘
```

zkML adds a **verification layer** before Crossmint payment execution. Invalid proof = transfer blocked.

---

## Why This Matters

### For CFOs
- **Delegate with confidence**: $50K/month autonomous agent spend with proof of every decision
- **Audit-ready**: Cryptographic receipts, not log files
- **Privacy preserved**: Prove compliance without revealing budget details

### For Compliance
- **SOX/SOC2**: Mathematical proof of policy adherence
- **EU AI Act**: Documented AI decision-making
- **Defensible audit trails**: Unforgeable, timestamped, linked to transactions

### For Developers
- **Simple integration**: Crossmint handles wallet complexity
- **Flexible verification**: Off-chain proof checking before payment
- **Multi-chain**: Deploy on any Crossmint-supported network

---

## Deployed Contracts (Base Sepolia)

| Contract | Address | Purpose |
|----------|---------|---------|
| ProofAttestation | [`0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213`](https://sepolia.basescan.org/address/0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213) | Stores proof hashes on-chain |
| SpendingGateWallet | [`0x6A47D13593c00359a1c5Fc6f9716926aF184d138`](https://sepolia.basescan.org/address/0x6A47D13593c00359a1c5Fc6f9716926aF184d138) | Gated transfers with proof verification |
| USDC (Testnet) | [`0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f`](https://sepolia.basescan.org/address/0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f) | Testnet stablecoin |

---

## Tech Stack

| Component | Role | Documentation |
|-----------|------|---------------|
| **Crossmint** | Enterprise MPC wallets, token transfers | [docs.crossmint.com](https://docs.crossmint.com) |
| **JOLT-Atlas** | zkML proof generation (~48KB SNARKs) | [github.com/ICME-Lab/jolt-atlas](https://github.com/ICME-Lab/jolt-atlas) |
| **Base Sepolia** | Testnet deployment, proof attestation | [sepolia.basescan.org](https://sepolia.basescan.org) |

---

## Key Files

```
src/
├── lib/
│   ├── crossmint.ts          # Crossmint API client
│   ├── baseSepolia.ts        # Base Sepolia contract integration
│   └── spendingModel.ts      # Enterprise policy model
├── hooks/
│   └── useCrossmintWallet.ts # React wallet hook
├── components/
│   └── CrossmintWalkthrough.tsx  # Interactive demo
└── app/api/crossmint/
    ├── wallet/route.ts       # Wallet creation endpoint
    └── transfer/route.ts     # Transfer + verification endpoint
```

---

## Crossmint Documentation

- [Wallets Overview](https://docs.crossmint.com/wallets/introduction)
- [Create Wallet API](https://docs.crossmint.com/api-reference/wallets/create-wallet)
- [Transfer Tokens](https://docs.crossmint.com/wallets/wallet-actions/transfer-tokens)
- [API Reference](https://docs.crossmint.com/api-reference/introduction)

---

## Summary

**Crossmint** provides enterprise wallet infrastructure with on-chain spending controls.
**zkML** extends verification to ML-based policies that can't run on-chain.
**Together**: The full spectrum of policy enforcement — from simple limits to complex ML models.

> Crossmint enforces rule-based policies on-chain. zkML proves ML-based policies were evaluated correctly. Together, they enable enterprise-grade agentic commerce.
