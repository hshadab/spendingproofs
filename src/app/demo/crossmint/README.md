# Verifiable Agentic Commerce: Crossmint × NovaNet zkML

An integration layer combining Crossmint's enterprise wallet infrastructure with NovaNet's zkML proofs for cryptographically verifiable AI spending decisions.

## The Enterprise Challenge

As AI agents gain spending authority, enterprises face a fundamental question: **How do you prove an agent followed policy, not just that it was authorized to spend?**

This matters because:

- **97% of CFOs understand AI agents can operate autonomously, but only 15% are deploying them** — the gap is trust, not technology ([PYMNTS 2025](https://www.pymnts.com/news/artificial-intelligence/2025/))
- **77% of AI-related issues result in direct financial loss** — enterprises need audit trails that can't be disputed ([ABA Banking Journal](https://bankingjournal.aba.com/2025/12/))
- **Regulatory frameworks are emerging** that require "proof traces identifying which rules passed or failed" for AI-influenced decisions ([AIVO Journal](https://www.aivojournal.org/))

## What zkML Adds

zkML (zero-knowledge machine learning) generates cryptographic proofs that a specific ML model ran on specific inputs and produced a specific output. For spending decisions, this means:

| Capability | What It Proves |
|------------|---------------|
| **Model Integrity** | The agent ran the exact policy model approved by the CFO |
| **Input Binding** | The proof is tied to this specific transaction's parameters |
| **Decision Verification** | The APPROVE/REJECT decision mathematically follows from the inputs |
| **Privacy Preservation** | Prove compliance without revealing budget limits, vendor scores, or policy thresholds |

This complements existing wallet infrastructure by adding a verification layer for the decision-making process itself.

---

## Demo Scenario: DataDog APM Subscription

An enterprise procurement agent evaluates a **$4,500/month DataDog APM** subscription against CFO-configured policies:

- **Monthly Budget**: $50,000
- **Max Per Transaction**: $10,000
- **Category Budget (Observability)**: $15,000
- **Max Vendor Risk**: 70%
- **Compliance Required**: Yes

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PROCUREMENT REQUEST EVALUATION                              │
│     Agent evaluates: DataDog APM @ $4,500/month                 │
│     Factors: Vendor risk (15%), History (92%), Compliance       │
│     Output: APPROVE/REJECT with confidence score                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. zkML POLICY PROOF GENERATION (Jolt-Atlas)                   │
│     SNARK proof generated (~48KB, ~10s)                         │
│     Proves: 6 vendor factors + 4 budget constraints evaluated   │
│     Privacy: Treasury balance, limits, vendor scores hidden     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. OFF-CHAIN VERIFICATION + PAYMENT (Crossmint)                │
│     Crossmint verifies SNARK proof OFF-CHAIN                    │
│     Proof valid? → Treasury executes USDC transfer              │
│     Payment authorized by cryptographic proof                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. AUDIT ATTESTATION (Arc Network)                             │
│     ProofAttestation contract records proof hash                │
│     Immutable audit trail for regulators/auditors               │
│     Attestation is for TRANSPARENCY, not payment gating         │
└─────────────────────────────────────────────────────────────────┘
```

## Enterprise Procurement Factors

The spending policy model evaluates multiple weighted factors:

| Factor | Example Value | Description |
|--------|---------------|-------------|
| `vendorRiskScore` | 0.15 (15%) | Risk assessment (lower is better) |
| `historicalVendorScore` | 0.92 (92%) | Historical performance (higher is better) |
| `vendorOnboardingDays` | 730 (2 years) | Relationship duration |
| `vendorComplianceStatus` | `true` | Passed compliance checks |
| `vendorTier` | `preferred` | Vendor classification |
| `categoryBudgetUsdc` | $15,000 | Budget for observability category |
| `categorySpentUsdc` | $4,200 | Already spent in category |

These factors are evaluated together — the proof guarantees all were considered, without revealing the specific thresholds or weights.

## Why This Matters

### The Liability Question

When an AI agent makes a spending decision, who bears responsibility if something goes wrong?

- **Without proof**: Disputes become "he said, she said" — logs can be questioned, policies can be misremembered
- **With zkML proof**: The cryptographic evidence shows exactly what policy ran and what decision it produced

### The Audit Question

Enterprises subject to SOX, GDPR, or industry regulations need defensible audit trails.

- **Traditional logs**: Can be modified, may not capture the full decision context
- **zkML proofs**: Mathematically unforgeable, tied to specific model versions and inputs

### The Trust Gap

The 97% → 15% gap (understand vs. deploy) exists because enterprises can't yet answer: "How do I know the agent did what it was supposed to do?"

zkML provides the answer: cryptographic proof of correct execution.

## Tech Stack

| Component | Role |
|-----------|------|
| **[Crossmint](https://docs.crossmint.com)** | Enterprise wallet infrastructure, MPC wallets, payment execution |
| **[Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas)** | zkML proof generation, SNARK proofs (~48KB) |
| **[Arc Network](https://arc.network)** | On-chain attestation for audit trail |

---

## Crossmint Integration Details

This demo uses three Crossmint APIs to create a secure, verifiable agentic commerce flow:

### 1. MPC Wallet Creation

**API**: [`POST /v1-alpha2/wallets`](https://docs.crossmint.com/wallets/quickstarts/create-wallets-api)

```typescript
// src/lib/crossmint.ts
const response = await fetch(`${CROSSMINT_API_URL}/v1-alpha2/wallets`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-KEY': CROSSMINT_SERVER_KEY,
  },
  body: JSON.stringify({
    type: 'evm-mpc-wallet',           // Fireblocks-backed MPC wallet
    linkedUser: 'zkml-demo-agent',    // Links wallet to agent identity
  }),
});
```

**Why MPC Wallets?**
- [Fireblocks-backed security](https://docs.crossmint.com/wallets/wallets/mpc-wallets) - enterprise-grade key management
- No private key exposure - keys are distributed across MPC nodes
- Idempotent creation - calling with same `linkedUser` returns existing wallet

### 2. zkML Proof Verification (Integration Layer)

Before executing any transfer, we verify the zkML proof cryptographically:

```typescript
// src/app/api/crossmint/transfer/route.ts
async function verifyProofBeforeTransfer(proof, modelId, modelHash, programIo) {
  const response = await fetch(`${PROVER_URL}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      proof,
      model_id: modelId,
      model_hash: modelHash,
      program_io: programIo,
    }),
  });

  const result = await response.json();
  return { valid: result.valid === true };
}
```

**This is where zkML adds value to Crossmint:**
- Proof verification happens BEFORE calling Crossmint Transfer API
- If proof is invalid → transfer is blocked (403 response)
- If proof is valid → proceed to Crossmint transfer
- The `proofHash` is included in transfer metadata for audit

### 3. Token Transfer Execution

**API**: [`POST /v1-alpha2/wallets/{locator}/tokens/{token}/transfers`](https://docs.crossmint.com/wallets/quickstarts/transfer-tokens-api)

```typescript
// src/lib/crossmint.ts
const response = await fetch(
  `${CROSSMINT_API_URL}/v1-alpha2/wallets/${walletLocator}/tokens/usdc/transfers`,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-KEY': CROSSMINT_SERVER_KEY,
    },
    body: JSON.stringify({
      recipient: `evm:${chain}:${toAddress}`,
      amount: amountUsdc.toString(),
      metadata: proofHash ? { zkmlProofHash: proofHash } : undefined,
    }),
  }
);
```

**Key Points:**
- Uses [Crossmint's token transfer infrastructure](https://docs.crossmint.com/wallets/quickstarts/transfer-tokens-api)
- Crossmint handles gas, signing, and execution
- `zkmlProofHash` in metadata creates audit trail linking payment to verified proof

### Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT (Next.js Frontend)                                                │
│  └── useCrossmintWallet hook                                             │
│      └── executeTransfer(to, amount, { proof, proofHash, programIo })    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  API ROUTE: POST /api/crossmint/transfer                                  │
│                                                                           │
│  Step 1: Verify zkML Proof                                                │
│  ┌────────────────────────────────┐                                       │
│  │ POST ${PROVER_URL}/verify      │ ──▶ JOLT-Atlas Prover                │
│  │ { proof, model_hash, ... }     │                                       │
│  └────────────────────────────────┘                                       │
│           │                                                               │
│           ▼                                                               │
│  ┌─ If proof INVALID ────────────┐                                        │
│  │ Return 403 PROOF_INVALID      │ ──▶ Transfer blocked                  │
│  └───────────────────────────────┘                                        │
│           │                                                               │
│           ▼                                                               │
│  Step 2: Execute Crossmint Transfer                                       │
│  ┌────────────────────────────────────────────┐                           │
│  │ POST /v1-alpha2/wallets/.../transfers      │ ──▶ Crossmint API        │
│  │ { recipient, amount, metadata: {proofHash} }│                          │
│  └────────────────────────────────────────────┘                           │
│           │                                                               │
│           ▼                                                               │
│  Step 3: Attest Proof Hash (Arc)                                          │
│  ┌────────────────────────────────┐                                       │
│  │ submitProofAttestation(hash)   │ ──▶ Arc ProofAttestation Contract    │
│  └────────────────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Supported Chains

| Chain | Transfer Method | Notes |
|-------|----------------|-------|
| Base Sepolia | Crossmint API | Full Crossmint integration |
| Polygon Amoy | Crossmint API | Full Crossmint integration |
| Base Mainnet | Crossmint API | Full Crossmint integration |
| Polygon | Crossmint API | Full Crossmint integration |
| Arbitrum | Crossmint API | Full Crossmint integration |
| Optimism | Crossmint API | Full Crossmint integration |
| Arc Testnet | Direct Transfer | Pending Crossmint sales enablement |

### Crossmint Documentation Links

- **Getting Started**: [docs.crossmint.com](https://docs.crossmint.com)
- **MPC Wallets Overview**: [docs.crossmint.com/wallets/wallets/mpc-wallets](https://docs.crossmint.com/wallets/wallets/mpc-wallets)
- **Create Wallets API**: [docs.crossmint.com/wallets/quickstarts/create-wallets-api](https://docs.crossmint.com/wallets/quickstarts/create-wallets-api)
- **Token Transfers API**: [docs.crossmint.com/wallets/quickstarts/transfer-tokens-api](https://docs.crossmint.com/wallets/quickstarts/transfer-tokens-api)
- **API Reference**: [docs.crossmint.com/api-reference](https://docs.crossmint.com/api-reference)

---

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
    ├── spendingModel.ts       # Enterprise procurement policy model
    └── arc.ts                 # Arc contract integration
```

## Arc Network Contracts

| Contract | Address | Purpose |
|----------|---------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` | Stablecoin transfers |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799952` | Records proof hashes for audit |

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

1. **Introduction** - The enterprise challenge and what zkML adds
2. **Agent Phase** - Procurement model evaluates $4,500 DataDog subscription
3. **Proof Phase** - Jolt-Atlas generates SNARK proof of policy compliance
4. **Wallet Phase** - Crossmint verifies proof off-chain, authorizes payment
5. **Execution** - USDC transfers, proof hash attested for audit trail
6. **Conclusion** - Verifiable enterprise procurement complete

## Regulatory Context

The trajectory is clear:

- **EU AI Act**: Requires documentation of AI decision-making processes
- **SEC guidance**: Increasing scrutiny on AI in financial operations
- **SOX implications**: AI-influenced spending decisions need audit trails

zkML provides the infrastructure to meet these requirements with cryptographic guarantees rather than policy assertions.

## References

- [PYMNTS: 2025 AI Agents in Payments](https://www.pymnts.com/news/artificial-intelligence/2025/)
- [ABA Banking Journal: Agentic AI Risks](https://bankingjournal.aba.com/2025/12/)
- [a16z: Know Your Agent](https://a16zcrypto.com/posts/article/big-ideas-crypto-2025/)
- [ICME: Definitive Guide to ZKML 2025](https://blog.icme.io/the-definitive-guide-to-zkml-2025/)

## Links

- [Crossmint](https://crossmint.com) - Enterprise wallet infrastructure
- [Jolt-Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML proof generation
- [Arc Network](https://arc.network) - On-chain attestation
- [Arc Explorer](https://testnet.arcscan.app) - View transactions
