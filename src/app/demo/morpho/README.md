# Morpho Blue zkML Spending Proofs Demo

**Trustless Autonomous DeFi: AI Agents Managing Morpho Vaults with Cryptographic Policy Compliance**

> This demo showcases NovaNet's Jolt-Atlas zkML infrastructure integrated with Morpho Blue.
> It connects to the **live zkML prover** and demonstrates policy-gated DeFi operations.

## Overview

This demo demonstrates how AI agents can autonomously manage Morpho Blue lending positions while proving every action complies with owner-defined spending policies through zkML proofs.

### The Problem

As AI agents become capable of managing DeFi positions, a critical trust gap emerges:
- Vault owners want to delegate management to AI agents for 24/7 optimization
- But they can't verify that agents will respect spending limits and risk parameters
- Traditional access controls are binary: either full access or none

### The Solution

**zkML Spending Proofs** require agents to generate cryptographic proofs for every operation:

```
Agent Decision → Jolt-Atlas SNARK → MorphoSpendingGate → Morpho Blue
                    (48KB proof)       (Verify)           (Execute)
```

**No proof, no transaction.**

## What You'll See

### 1. Policy Configuration
Define spending constraints for the AI agent:
- **Daily Limits**: Maximum capital deployment per day
- **Single Transaction Limits**: Max amount per operation
- **LTV Bounds**: Maximum loan-to-value ratio
- **Health Factor**: Minimum collateralization safety margin
- **Market Whitelist**: Allowed lending markets

### 2. Agent Operations
Watch the agent make decisions and generate proofs for:
- **Supply**: Deploying funds into lending markets
- **Borrow**: Taking collateralized loans
- **Withdraw**: Moving funds out of markets
- **Repay**: Clearing debt positions

### 3. Live Proof Generation
Toggle between:
- **Real Mode**: Calls live NovaNet prover (4-12s proving time)
- **Simulated Mode**: Mock proofs for faster demos

## Technical Details

| Metric | Value |
|--------|-------|
| Proof Size | ~48KB |
| Proving Time | 4-12s (warm) / ~30s (cold) |
| Verification Gas | ~200K |
| Chain | Arc Testnet (ID: 5042002) |

### Deployed Contracts

| Contract | Address |
|----------|---------|
| MockMorphoBlue | `0x034459863E9d2d400E4d005015cB74c2Cd584e0E` |
| MorphoSpendingGate | `0x93BDD317371A2ab0D517cdE5e9FfCDa51247770D` |

## SDK Integration

```typescript
import { createMorphoSpendingProofsClient, MorphoOperation } from '@/sdk/morpho';

const client = createMorphoSpendingProofsClient({
  proverUrl: 'https://arc-policy-proofs.onrender.com',
  chainId: 5042002,
  gateAddress: '0x93BDD317371A2ab0D517cdE5e9FfCDa51247770D',
  morphoAddress: '0x034459863E9d2d400E4d005015cB74c2Cd584e0E',
});

const proof = await client.generateProof({
  policy,
  operation: MorphoOperation.SUPPLY,
  amount: 1000000000n,
  market: '0x...',
  agent: '0x...',
}, signer);
```

## What's Real vs. Simulated

### Real Components
- zkML proof generation (live Jolt-Atlas prover)
- Market data from Morpho API
- Policy structure and validation
- Demo UI workflow

### Simulated Components
- On-chain transactions (demo shows flow, no actual signing)
- Agent decisions (pre-defined sequence)
- On-chain verification (off-chain for demo)

## Related Resources

- [Main Spending Proofs Documentation](/docs/ARCHITECTURE.md)
- [API Reference](/docs/API.md)
- [Morpho Blue Documentation](https://docs.morpho.org)
- [Arc Testnet Explorer](https://testnet.arcscan.app)

---

**Part of the [NovaNet Jolt-Atlas Spending Proofs](/) project.**
