# Architecture

This document describes the system architecture of Spending Policy Proofs.

## Overview

Spending Policy Proofs is a cryptographic infrastructure for autonomous agent spending control on Arc Network. It uses zero-knowledge machine learning (zkML) proofs to verify that agents follow their spending policies before transactions are executed.

```
                                    ARCHITECTURE OVERVIEW
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Client    │───▶│  Next.js    │───▶│   Prover    │───▶│     Arc Chain       │  │
│  │   (SDK)     │    │   API       │    │ (Jolt-Atlas)│    │   (Settlement)      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## System Components

### 1. Client SDK (`@hshadab/spending-proofs`)

The SDK provides a TypeScript interface for applications to integrate spending proofs.

**Location:** `/sdk/`

**Key Files:**
- `src/index.ts` - Main SDK entry point
- `src/client.ts` - PolicyProofs client class
- `src/wallet.ts` - Wallet integration helpers
- `src/react.ts` - React hooks for frontend apps
- `src/wagmi.ts` - Wagmi connector integration

**Exports:**
- `PolicyProofs` - Main client class
- `useSpendingProofs` - React hook
- Wagmi connector configuration

### 2. Next.js Application

The web application provides demo UI and API proxy routes.

**Location:** `/src/`

**Structure:**
```
src/
├── app/                    # Next.js App Router pages
│   ├── api/               # API routes (proxy to prover)
│   │   ├── prove/         # Proof generation endpoint
│   │   ├── crossmint/     # Crossmint wallet integration
│   │   └── circle/        # Circle integration
│   └── demo/              # Interactive demo pages
├── components/            # React components
│   ├── landing/           # Marketing page components
│   └── diagrams/          # Architecture visualizations
├── hooks/                 # React hooks
├── lib/                   # Core libraries
└── __tests__/            # Unit tests
```

### 3. Prover Service (Jolt-Atlas)

The Rust-based zkML prover generates SNARK proofs for spending decisions.

**Location:** `/prover/`

**Key Components:**
- ONNX neural network inference
- Jolt SNARK proof generation
- REST API for proof requests

**Performance:**
- Warm proof generation: 4-12 seconds
- Cold start: ~30 seconds
- Proof size: ~48KB

### 4. Smart Contracts

Solidity contracts deployed on Arc Testnet for on-chain enforcement.

**Location:** `/contracts/`

**Contracts:**
- `ProofAttestation.sol` - Records proof hashes on-chain
- `SpendingGateWallet.sol` - Enforced spending limits with proof verification
- `TestnetUSDC.sol` - Test token for Arc testnet

## Data Flow

### Proof Generation Flow

```
1. Client prepares spending input
   ↓
2. SDK calls /api/prove endpoint
   ↓
3. API proxy forwards to Jolt-Atlas prover
   ↓
4. Prover runs ONNX inference
   ↓
5. Prover generates SNARK proof
   ↓
6. Proof hash returned to client
   ↓
7. Optional: Submit attestation to Arc chain
```

### Transfer Flow (with Enforcement)

```
1. Client generates proof for transaction intent
   ↓
2. Proof attested on ProofAttestation contract
   ↓
3. Call gatedTransfer on SpendingGateWallet
   ↓
4. Contract verifies proof is attested and unused
   ↓
5. Contract executes USDC transfer
   ↓
6. Proof marked as used (prevents replay)
```

## Key Libraries

### `/src/lib/` Structure

| File | Purpose |
|------|---------|
| `config.ts` | Centralized configuration (addresses, env vars) |
| `crypto.ts` | Secure random generation utilities |
| `validation.ts` | Input validation for API routes |
| `metrics.ts` | Structured logging and metrics |
| `retry.ts` | Exponential backoff retry logic |
| `proofCache.ts` | LRU cache with version invalidation |
| `signatureAuth.ts` | Wallet signature verification |
| `spendingModel.ts` | Spending decision model and validation |
| `arc.ts` | Arc Network contract interactions |
| `crossmint.ts` | Crossmint wallet API client |
| `errors.ts` | Typed error classes |
| `types.ts` | TypeScript type definitions |

## Security Model

### Proof Verification

1. **Off-chain verification** - Proofs verified by examining proof structure and hashes
2. **On-chain attestation** - Proof hashes recorded on ProofAttestation contract
3. **Enforcement** - SpendingGateWallet requires attested proof for transfers

### Input Binding

Proofs are bound to transaction intents via:
- Recipient address
- Transfer amount
- Nonce (prevents replay)
- Expiry timestamp
- Policy ID

### Replay Protection

- Each proof can only be used once
- SpendingGateWallet tracks used proof hashes
- Proofs expire after configurable duration

## Deployment Architecture

### Production Setup

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Vercel/CF      │────▶│    Render        │────▶│   Arc Testnet    │
│   (Next.js)      │     │  (Jolt Prover)   │     │   (Contracts)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Environment Variables

See `.env.example` for required configuration:
- `NEXT_PUBLIC_ARC_RPC` - Arc testnet RPC
- `PROVER_BACKEND_URL` - Jolt-Atlas prover URL
- `CROSSMINT_SERVER_KEY` - Crossmint API key (server-side)

## Future Architecture

### Arc Native Precompile

Future integration will move proof verification into Arc's execution layer:

```
┌─────────────────────────────────────────────────────────────────┐
│                       ARC CHAIN                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Smart       │───▶│ Precompile  │───▶│    State            │ │
│  │ Contract    │    │ (Verifier)  │    │    Changes          │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

This will enable:
- On-chain proof verification (not just attestation)
- Lower gas costs for verification
- Tighter integration with Arc's consensus

See [precompile-spec.md](./precompile-spec.md) for specification.
