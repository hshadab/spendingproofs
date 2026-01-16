# Extending Skyfire with zkML Spending Policy Verification

**zkML spending policy proofs** powered by [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) — adding cryptographic verification of agent decision-making to Skyfire's identity and payment infrastructure.

---

## Overview

Skyfire provides:
- **KYA (Know Your Agent)** — Verified agent identity via JWT tokens
- **PAY Tokens** — Payment authorization with merchant binding
- **Agent Registry** — Directory of verified agents and services

This demo adds a verification layer:
- **"Did the agent follow its spending policy?"** → Cryptographic proof (not logs, not attestations from the agent itself)

---

## Architecture

```
Skyfire KYA              zkML Proof                Arc Attestation
    │                        │                          │
    ▼                        ▼                          ▼
┌──────────┐          ┌──────────┐              ┌──────────┐
│ Verified │          │ Policy   │              │ On-Chain │
│ Agent ID │    →     │ Proof    │      →       │ Record   │
│ (JWT)    │          │ (SNARK)  │              │ (Tx)     │
└──────────┘          └──────────┘              └──────────┘
     │                      │                        │
     └──────────────────────┴────────────────────────┘
                            │
         Skyfire PAY Token + zkML Verification Hash
                            │
                    ┌──────────────┐
                    │  Gated       │
                    │  Transfer    │
                    └──────────────┘
```

### Flow

1. **Identity** — Skyfire KYA establishes verified agent identity
2. **Policy Evaluation** — Agent runs spending policy model (vendor risk, budget, compliance)
3. **Proof Generation** — JOLT-Atlas generates ~48KB SNARK proving correct evaluation
4. **Binding** — `verificationHash = keccak256(proofHash, agentId, decision, confidence, timestamp)`
5. **Attestation** — Record verification on-chain (Arc testnet)
6. **Payment** — Skyfire PAY token authorizes transfer; smart contract checks attestation

---

## Skyfire API Usage

### KYA+PAY Token Creation

```typescript
const token = await fetch('https://api-sandbox.skyfire.xyz/api/v1/tokens', {
  method: 'POST',
  headers: { 'skyfire-api-key': apiKey },
  body: JSON.stringify({
    type: 'kya+pay',
    buyerTag: `zkml-verified-${proofHash.slice(0, 8)}`,
    sellerServiceId: serviceId,
    tokenAmount: amount,
    expiresAt: Math.floor(Date.now() / 1000) + 3600
  })
});
```

### Proof-Identity Binding

```typescript
// Cryptographically bind zkML proof to Skyfire agent ID
const verificationHash = keccak256(encodeAbiParameters(
  parseAbiParameters('bytes32, string, bool, uint256, uint256'),
  [proofHash, agentId, decision, confidence, timestamp]
));
```

---

## Spending Policy Model

6-factor policy evaluation:

```typescript
interface SpendingPolicyInput {
  purchaseAmount: number;      // Amount to spend
  availableBudget: number;     // Total budget limit
  vendorRiskScore: number;     // 0-1 (lower = safer)
  vendorTrackRecord: number;   // 0-1 (higher = better)
  categoryBudgetLeft: number;  // Category allocation
  complianceStatus: boolean;   // Vendor compliance verified
}

// Output: APPROVE/REJECT with confidence score
```

The model runs off-chain. JOLT-Atlas generates a SNARK proving it ran correctly on the given inputs.

---

## Performance

| Metric | Value |
|--------|-------|
| zkML Proof Size | ~48KB |
| Proof Generation | 4-12 seconds |
| Proof Verification | <150ms |
| Skyfire KYA Token | ~300ms |
| Skyfire PAY Token | ~400ms |

---

## Running the Demo

### Demo Mode (No API Key)
```bash
npm run dev
# Visit /demo/skyfire
```
- Simulated Skyfire tokens (realistic JWT format)
- Real zkML proof generation
- Mock transactions

### Live Mode
```bash
# .env.local
SKYFIRE_API_KEY=your-api-key
SKYFIRE_API_URL=https://api-sandbox.skyfire.xyz
NEXT_PUBLIC_SKYFIRE_ENABLED=true
```
- Real Skyfire API calls
- Real on-chain attestations (Arc testnet)
- Real USDC transfers via SpendingGateWallet

---

## Contracts (Arc Testnet)

| Contract | Address |
|----------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` |

---

## Code Structure

```
src/
├── lib/skyfire/
│   ├── client.ts        # Skyfire API client
│   ├── config.ts        # Configuration
│   └── types.ts         # TypeScript interfaces
├── app/api/skyfire/
│   ├── agent/route.ts   # POST - Create agent with KYA
│   ├── pay-token/route.ts  # POST - Generate PAY token
│   └── transfer/route.ts   # POST - Execute verified transfer
└── components/skyfire/
    └── SkyfireWalkthrough.tsx
```

---

## API Endpoints

### This Demo

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/skyfire/agent` | POST | Create agent, generate KYA token |
| `/api/skyfire/pay-token` | POST | Generate PAY token with proof binding |
| `/api/skyfire/transfer` | POST | Full flow: KYA → proof → attestation → transfer |

### Skyfire API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tokens` | POST | Create KYA, PAY, or KYA+PAY tokens |
| `/api/v1/tokens/introspect` | POST | Validate and decode tokens |
| `/api/v1/directory/services` | GET | List available services |

---

## Potential Extensions

Ideas for native Skyfire integration:

1. **PAY Token Fields** — Optional `proofHash` and `verificationHash` in token payload
2. **Verification Service** — Skyfire-hosted zkML verification before payment authorization
3. **Policy Registry** — Agents publish spending policies; Skyfire validates proofs match
4. **Compliance API** — Query verification status for audit

---

## Resources

- [Skyfire Docs](https://docs.skyfire.xyz)
- [Skyfire Sandbox](https://app-sandbox.skyfire.xyz)
- [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas)
- [Arc Testnet](https://testnet.arcscan.app)

---

## License

MIT — Part of the [Spending Proofs](https://github.com/hshadab/spendingproofs) project.
