# Verifiable Agentic Commerce: Crossmint + zkML

Combining **Crossmint's enterprise wallet infrastructure** with **zkML policy proofs** for cryptographically verifiable AI spending decisions.

## The Problem: Smart Contracts Have a Complexity Ceiling

Smart contracts excel at simple rules:
```solidity
require(amount <= limit);  // ✓ Works great
```

But enterprise spending policies require:
- ML-based vendor risk scoring
- Historical performance analysis
- Multi-factor approval matrices
- Category budget tracking
- Compliance verification

```solidity
require(mlModel.evaluate(vendorRisk, history, budget, compliance) == APPROVE);  // ✗ Infeasible
```

**This computation is infeasible on-chain.** Gas costs explode with complexity. ML inference cannot run in the EVM.

## The Solution: zkML + Crossmint

**zkML** bridges this gap:
- Run complex policy models **off-chain**
- Generate cryptographic proofs of correct execution
- Verify proofs **on-chain** (or off-chain before payment)

**Crossmint** provides the enterprise wallet infrastructure:
- MPC wallets with Fireblocks-backed security
- Token transfer APIs with metadata for audit trails
- Gas abstraction and multi-chain support

**Together**: Complex AI decisions + Cryptographic verification + Enterprise-grade payments

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

**Crossmint** provides the enterprise wallet infrastructure.
**zkML** provides cryptographic proof of policy compliance.
**Together**: Verifiable agentic commerce that CFOs can trust.

> Smart contracts check 1 factor. Our policy model checks 6. zkML proves all 6 were evaluated — and Crossmint executes the payment.
