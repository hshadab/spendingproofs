# Extending Skyfire with zkML Spending Policy Verification

**A demonstration of how zkML can complement Skyfire's agent identity and payment infrastructure**

---

## For Skyfire

This demo explores how **zkML spending policy proofs** powered by [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) could extend Skyfire's capabilities — adding cryptographic verification of agent decision-making to Skyfire's already robust identity and payment infrastructure.

### What Skyfire Provides

Skyfire already delivers powerful agent commerce capabilities:
- **KYA (Know Your Agent)** — Verified agent identity via JWT tokens
- **PAY Tokens** — Secure payment authorization with merchant binding
- **Agent Registry** — Trusted directory of verified agents and services
- **Payment Rails** — Seamless agent-to-agent and agent-to-service payments

### What zkML Adds

zkML extends Skyfire by answering a third question enterprises are asking:
- **"Did the agent follow its spending policy?"** → Cryptographic proof

This complements Skyfire's identity layer with **provable policy compliance** — not logs or agent self-attestations, but mathematical proof.

---

## Why This Matters

As AI agents gain spending authority through platforms like Skyfire, enterprises want additional verification layers for high-value or regulated transactions. zkML provides this without changing Skyfire's core architecture.

### The Complete Picture

| Layer | Skyfire Provides | zkML Extends |
|-------|------------------|--------------|
| **Identity** | KYA tokens — verified agent identity | Binds proofs to verified agent ID |
| **Authorization** | PAY tokens — payment permissions | Adds policy compliance verification |
| **Audit Trail** | Transaction records | On-chain proof attestations |
| **Enforcement** | Service-level controls | Smart contract proof gates |

---

## What This Demo Shows

### Skyfire Components Used

| Component | How We Use It |
|-----------|---------------|
| **KYA Tokens** | Agent identity verification via `POST /api/v1/tokens` |
| **KYA+PAY Tokens** | Combined identity and payment authorization |
| **Agent Registry** | Service discovery via Skyfire directory |
| **JWT Claims** | Agent ID, issuer, expiration extracted and displayed |

### What zkML Adds

| Component | What It Does |
|-----------|--------------|
| **Spending Policy Model** | ONNX neural network evaluating vendor risk, budget, compliance |
| **zkML Policy Proof** | ~48KB SNARK proving policy evaluated correctly |
| **Proof-Agent Binding** | `verificationHash = hash(proofHash, agentId, decision, confidence, timestamp)` |
| **On-Chain Attestation** | Verification result recorded immutably on Arc testnet |
| **SpendingGateWallet** | Smart contract enforcing "no attestation = no funds" |

### The Complete Flow

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

---

## Integration Architecture

### How zkML Extends Skyfire

1. **Identity Layer** — Skyfire KYA establishes verified agent identity (unchanged)
2. **Policy Layer** — zkML proves spending policy was correctly evaluated (new)
3. **Binding Layer** — `verificationHash` cryptographically links proof to Skyfire agent ID (new)
4. **Attestation Layer** — On-chain record of verification for audit (new)
5. **Payment Layer** — Skyfire PAY authorizes transfer with proof linkage (extended)

### Value for the Skyfire Ecosystem

| Skyfire Today | Skyfire + zkML |
|---------------|----------------|
| Proves agent identity (KYA) | Proves agent identity (KYA) |
| Authorizes payments (PAY) | Authorizes payments (PAY) |
| Transaction records | Transaction records |
| — | **Proves policy was followed** |
| — | **On-chain enforcement option** |
| Audit: who + what | Audit: who + what + **verified how** |

### Enterprise Use Cases

- **Financial Services**: Prove AI trading agents followed risk policies before Skyfire payment
- **Procurement**: Verify purchasing agents checked vendor compliance via zkML, pay via Skyfire
- **Treasury**: Confirm agents stayed within budget allocations with cryptographic proof
- **Compliance**: Auditable verification trail for regulators (Skyfire identity + zkML policy)

---

## Technical Implementation

### Skyfire Integration

```typescript
// Create Skyfire agent with KYA identity
const agent = await createAgent('zkML Demo Agent');

// Generate KYA+PAY token for payment
const token = await fetch('/api/v1/tokens', {
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

### zkML Proof Binding

```typescript
// Bind zkML proof to Skyfire agent identity
const verificationHash = keccak256(encodeAbiParameters(
  parseAbiParameters('bytes32, string, bool, uint256, uint256'),
  [proofHash, agentId, decision, confidence, timestamp]
));

// Include in PAY token metadata
const payToken = await generatePayToken({
  agentId: agent.id,
  amount,
  proofHash,           // zkML proof reference
  verificationHash,    // Cryptographic binding
});
```

### Spending Policy Model

The demo uses a simple 6-factor spending policy:

```typescript
interface SpendingPolicyInput {
  purchaseAmount: number;      // Amount to spend
  availableBudget: number;     // Total budget limit
  vendorRiskScore: number;     // 0-1 (lower = safer)
  vendorTrackRecord: number;   // 0-1 (higher = better)
  categoryBudgetLeft: number;  // Category allocation
  complianceStatus: boolean;   // Vendor compliance verified
}

// Model outputs APPROVE/REJECT with confidence score
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| zkML Proof Size | ~48KB |
| Proof Generation | 4-12 seconds |
| Proof Verification | <150ms |
| Skyfire KYA Token | ~300ms |
| Skyfire PAY Token | ~400ms |

---

## Demo Modes

### Demo Mode (No API Key Required)
- Simulated Skyfire tokens (realistic JWT format)
- Real zkML proof generation (JOLT-Atlas)
- Mock transaction hashes
- Full flow demonstration

### Live Mode (Real Skyfire API)
- Real Skyfire KYA token generation
- Real Skyfire PAY token creation
- Real on-chain attestation tx
- Real gated USDC transfer on Arc testnet
- Decoded JWT claims displayed

---

## Running the Demo

### Quick Start (Demo Mode)
1. Visit the demo page
2. Click "Start Demo" or step through manually
3. Watch the 5-phase flow execute

### Live Mode (Real Transactions)
1. Configure Skyfire API key in `.env.local`:
   ```bash
   SKYFIRE_API_KEY=your-api-key
   SKYFIRE_API_URL=https://api-sandbox.skyfire.xyz
   NEXT_PUBLIC_SKYFIRE_ENABLED=true
   ```
2. Connect wallet to Arc Testnet (Chain ID: 5042002)
3. Get testnet USDC from [faucet.circle.com](https://faucet.circle.com)
4. Run the demo with real Skyfire API calls

---

## Contracts (Arc Testnet)

| Contract | Address | Purpose |
|----------|---------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` | Token transfers |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` | Verification attestations |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` | Gated transfers |

---

## Code Structure

```
src/
├── lib/skyfire/
│   ├── client.ts        # Skyfire API client
│   ├── config.ts        # API configuration
│   └── types.ts         # TypeScript interfaces
├── app/api/skyfire/
│   ├── agent/route.ts   # Agent creation endpoint
│   ├── pay-token/route.ts  # PAY token generation
│   └── transfer/route.ts   # Complete verified transfer
├── components/skyfire/
│   └── SkyfireWalkthrough.tsx  # Main demo component
└── lib/arc.ts           # Arc contract integration
```

---

## API Endpoints

### Internal API (This Demo)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/skyfire/agent` | POST | Create Skyfire agent with KYA token |
| `/api/skyfire/pay-token` | POST | Generate PAY token with proof binding |
| `/api/skyfire/transfer` | POST | Execute complete verified transfer |

### Skyfire API Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tokens` | POST | Create KYA, PAY, or KYA+PAY tokens |
| `/api/v1/tokens/introspect` | POST | Validate and decode tokens |
| `/api/v1/directory/services` | GET | List available services |

---

## Discussion Points

### Potential Integration Paths

1. **PAY Token Extension**: Add optional `proofHash` and `verificationHash` fields to PAY tokens
2. **Policy Verification Service**: Skyfire-hosted zkML verification before payment authorization
3. **Agent Policy Registry**: Allow agents to publish their spending policies
4. **Compliance API**: Query zkML verification status for audit purposes

### What We'd Love to Explore

- Feedback on the KYA + zkML binding pattern
- Interest in native policy verification support in Skyfire
- Potential for zkML as an optional Skyfire extension for regulated industries

---

## Resources

- [Skyfire Documentation](https://docs.skyfire.xyz) — API Reference
- [Skyfire Sandbox](https://app-sandbox.skyfire.xyz) — Test Dashboard
- [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) — zkML Prover
- [Arc Network](https://arc.network) — Testnet for attestations

---

## Contact

Interested in discussing zkML + Skyfire integration? We'd love to connect.

**This demo is part of the [Spending Proofs](https://github.com/hshadab/spendingproofs) project** — exploring verifiable AI agent spending policies.
