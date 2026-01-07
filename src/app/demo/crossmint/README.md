# Crossmint Integration Demo

An interactive walkthrough showing how Crossmint enterprise wallets integrate with [Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas) zkML spending proofs for trustless AI agent commerce.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. AI AGENT DECISION                                           │
│     Agent evaluates purchase: Weather API @ $0.05, 98% uptime   │
│     Runs spending model locally against policy constraints      │
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
│  3. PROOF ATTESTATION (Arc Network)                             │
│     ProofAttestation contract records proof hash                │
│     SpendingGate checks: is this proof attested?                │
│     No attestation = transaction reverts                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. PAYMENT EXECUTION (Crossmint)                               │
│     MPC Wallet (Fireblocks-backed) signs transaction            │
│     USDC transfers on Arc Network                               │
│     Proof hash recorded for audit trail                         │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Role | Product |
|-----------|------|---------|
| **Crossmint** | Wallet infrastructure | MPC Wallets, Agentic Commerce, Headless Checkout |
| **[Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas)** | Proof generation | SNARK proofs (~48KB) |
| **Arc Network** | On-chain attestation | ProofAttestation + SpendingGate contracts |

## Key Files

```
src/
├── app/
│   ├── api/crossmint/
│   │   ├── wallet/route.ts    # Crossmint wallet API
│   │   └── transfer/route.ts  # USDC transfer with proof attestation
│   └── demo/crossmint/
│       └── page.tsx           # Demo entry point
├── components/
│   └── CrossmintWalkthrough.tsx  # Annotated walkthrough UI
├── hooks/
│   └── useCrossmintWallet.ts  # Wallet state management
└── lib/
    ├── crossmint.ts           # Crossmint API client
    └── arc.ts                 # Arc contract integration
```

## Arc Network Contracts

| Contract | Address | Purpose |
|----------|---------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` | Stablecoin transfers |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` | Records proof hashes |
| SpendingGate | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` | Enforces attestation before transfer |

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
2. **Agent Phase** - AI agent evaluates a $0.05 Weather API purchase
3. **Proof Phase** - Jolt-Atlas generates SNARK proof of policy compliance
4. **Wallet Phase** - Crossmint wallet receives proof for attestation
5. **Execution** - USDC transfers after proof is attested on Arc
6. **Conclusion** - Trustless commerce complete

## Why This Matters

Current AI agent wallets rely on:
- Rate limits
- Allowlists
- Trust in agent code

This demo shows a better approach:
- **Cryptographic proof** that agent followed spending policies
- **On-chain attestation** before funds can move
- **No trust required** - only math

## Links

- [Crossmint](https://crossmint.com) - Enterprise wallet infrastructure
- [Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML proof generation
- [Arc Network](https://arc.network) - On-chain attestation
- [Arc Explorer](https://testnet.arcscan.app) - View transactions
