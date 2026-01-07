# Crossmint Integration Demo

An interactive walkthrough showing how Crossmint enterprise wallets integrate with [Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas) zkML spending proofs for trustless AI agent commerce.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. AI AGENT DECISION (Real Model)                              │
│     Agent evaluates purchase: Weather API @ $0.05, 98% uptime   │
│     Runs REAL spending model locally against policy constraints │
│     Output: APPROVE/REJECT with confidence score                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. zkML PROOF GENERATION (Jolt-Atlas)                          │
│     SNARK proof generated (~48KB, ~10s)                         │
│     Proves: model ran correctly, output = APPROVE               │
│     Privacy: Treasury balance, limits stay hidden               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. OFF-CHAIN VERIFICATION + PAYMENT (Crossmint)                │
│     Crossmint verifies SNARK proof OFF-CHAIN                    │
│     Proof valid? → MPC Wallet executes USDC transfer            │
│     Payment authorized by cryptographic proof, not attestation  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. AUDIT ATTESTATION (Arc Network)                             │
│     ProofAttestation contract records proof hash                │
│     Creates immutable audit trail for compliance                │
│     Attestation is for TRANSPARENCY, not payment gating         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Insight

**Payment is authorized by OFF-CHAIN proof verification.**

On-chain attestation is for audit trail and transparency - it does NOT gate payments. This is the same model used by the main spendingproofs demo.

## Tech Stack

| Component | Role | Product |
|-----------|------|---------|
| **[Crossmint](https://docs.crossmint.com)** | Wallet infrastructure | MPC Wallets, Agentic Commerce, Headless Checkout |
| **[Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas)** | Proof generation | SNARK proofs (~48KB) |
| **[Arc Network](https://docs.arc.net)** | Audit attestation | ProofAttestation contract for audit trail |

## Key Files

```
src/
├── app/
│   ├── api/crossmint/
│   │   ├── wallet/route.ts    # Crossmint wallet API
│   │   └── transfer/route.ts  # USDC transfer + audit attestation
│   └── demo/crossmint/
│       └── page.tsx           # Demo entry point
├── components/
│   └── CrossmintWalkthrough.tsx  # Annotated walkthrough UI
├── hooks/
│   └── useCrossmintWallet.ts  # Wallet state management
└── lib/
    ├── crossmint.ts           # Crossmint API client
    ├── spendingModel.ts       # REAL spending decision model
    └── arc.ts                 # Arc contract integration
```

## Arc Network Contracts

| Contract | Address | Purpose |
|----------|---------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` | Stablecoin transfers |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` | Records proof hashes for audit |
| SpendingGate | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` | Optional enforcement (not used for payment gating) |

## Environment Variables

```bash
# Crossmint API (get from crossmint.com)
CROSSMINT_SERVER_KEY=sk_production_...
CROSSMINT_CLIENT_KEY=ck_production_...
CROSSMINT_API_URL=https://www.crossmint.com/api

# Demo wallet for Arc transfers
DEMO_WALLET_PRIVATE_KEY=0x...
```

## The Demo Flow

1. **Introduction** - Overview of Crossmint + zkML integration
2. **Agent Phase** - REAL spending model evaluates $0.05 Weather API purchase
3. **Proof Phase** - Jolt-Atlas generates SNARK proof of policy compliance
4. **Wallet Phase** - Crossmint verifies proof OFF-CHAIN, authorizes payment
5. **Execution** - USDC transfers, proof hash attested for audit trail
6. **Conclusion** - Trustless commerce complete

## Why This Matters

Current AI agent wallets rely on:
- Rate limits
- Allowlists
- Trust in agent code

This demo shows a better approach:
- **Real spending model** runs actual policy logic (not simulated)
- **Cryptographic proof** that agent followed spending policies
- **Off-chain verification** authorizes payments instantly
- **On-chain attestation** provides immutable audit trail
- **No trust required** - only math

## Links

- [Crossmint](https://crossmint.com) - Enterprise wallet infrastructure
- [Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML proof generation
- [Arc Network](https://arc.network) - On-chain attestation
- [Arc Explorer](https://testnet.arcscan.app) - View transactions
