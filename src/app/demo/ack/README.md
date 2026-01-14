# Extending ACK with zkML Spending Policy Verification

**A demonstration of how zkML can complement Catena's Agent Commerce Kit**

---

## For Catena Labs

This demo explores how **zkML spending policy proofs** could extend ACK's capabilities — adding cryptographic verification of agent decision-making to ACK's existing identity and receipt infrastructure.

### The Opportunity

ACK already answers the critical questions:
- **"Who is this agent?"** → ACK-ID (W3C DIDs)
- **"What payments were made?"** → ACK-Pay (Verifiable Credentials)

zkML adds a third question enterprises are asking:
- **"Did the agent follow its spending policy?"** → Cryptographic proof

### Why This Matters for ACK

As AI agents gain spending authority, enterprises need more than identity and receipts — they need **proof of policy compliance**. Not logs. Not attestations from the agent itself. Mathematical proof.

This is a natural extension of ACK's mission: making AI agent commerce verifiable and auditable.

---

## What This Demo Shows

### ACK Components Used

| Component | How We Use It |
|-----------|---------------|
| **ACK-ID** | Agent identity via `did:key`, ControllerCredential linking agent to owner wallet |
| **ACK-Pay** | PaymentReceiptCredential issued after successful transfer, linking DID + proof + tx |
| **W3C Standards** | DIDs, Verifiable Credentials, EIP-712 signatures |

### What zkML Adds

| Component | What It Does |
|-----------|--------------|
| **Spending Policy Model** | ONNX neural network evaluating vendor risk, budget, compliance |
| **zkML Policy Proof** | ~48KB SNARK proving policy evaluated correctly |
| **On-Chain Attestation** | Verification result recorded immutably |
| **SpendingGateWallet** | Smart contract enforcing "no attestation = no funds" |

### The Complete Flow

```
ACK-ID                    zkML                      ACK-Pay
   │                        │                          │
   ▼                        ▼                          ▼
┌──────────┐          ┌──────────┐              ┌──────────┐
│ Agent    │          │ Policy   │              │ Payment  │
│ Identity │    →     │ Proof    │      →       │ Receipt  │
│ (DID)    │          │ (SNARK)  │              │ (VC)     │
└──────────┘          └──────────┘              └──────────┘
     │                      │                        │
     └──────────────────────┴────────────────────────┘
                            │
                    Complete Audit Trail
```

---

## Integration Concept

### How It Could Work with ACK

1. **Identity Layer** — ACK-ID establishes verifiable agent identity (unchanged)
2. **Policy Layer** — zkML proves spending policy was correctly evaluated (new)
3. **Enforcement Layer** — Smart contract checks proof before releasing funds (new)
4. **Receipt Layer** — ACK-Pay issues receipt with proof linkage (extended)

### Value Add for ACK Ecosystem

| ACK Today | ACK + zkML |
|-----------|------------|
| Proves agent identity | Proves agent identity |
| Proves payment occurred | Proves payment occurred |
| — | **Proves policy was followed** |
| — | **On-chain enforcement** |
| Audit trail: who + what | Audit trail: who + what + **verified how** |

### Enterprise Use Cases

- **Financial Services**: Prove AI trading agents followed risk policies
- **Procurement**: Verify purchasing agents checked vendor compliance
- **Treasury**: Confirm agents stayed within budget allocations
- **Compliance**: Auditable proof of policy adherence for regulators

---

## Technical Implementation

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

### zkML Proof Generation

| Metric | Value |
|--------|-------|
| Proof Size | ~48KB |
| Generation | 4-12 seconds |
| Verification | <150ms |
| Prover | JOLT-Atlas |

### ACK-Pay Receipt Extension

The PaymentReceiptCredential includes proof linkage:

```typescript
interface ExtendedACKPaymentReceipt {
  // Standard ACK-Pay fields
  receiptCredential: VerifiableCredential;
  txHash: string;
  amount: string;

  // zkML extensions
  proofHash: string;           // Links to zkML proof
  attestationTxHash: string;   // On-chain verification record
  verificationHash: string;    // hash(proof + decision + timestamp)
}
```

---

## Demo Modes

### Demo Mode (No Wallet Required)
- Simulated wallet address
- Real zkML proof generation (JOLT-Atlas)
- Mock transaction hashes
- Full flow demonstration

### Live Mode (Real Transactions)
- Connect wallet to Arc Testnet
- Real EIP-712 credential signatures
- Real on-chain attestation tx
- Real gated USDC transfer
- Real PaymentReceiptCredential

---

## Running the Demo

### Quick Start (Demo Mode)
1. Visit the demo page
2. Click "Play Demo" or step through manually
3. Watch the 6-phase flow execute

### Live Mode
1. Connect wallet to Arc Testnet (Chain ID: 5042002)
2. Get testnet USDC from [faucet.circle.com](https://faucet.circle.com)
3. Toggle to "Live" mode
4. Run the demo with real transactions

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
├── lib/ack/
│   ├── client.ts         # ACK SDK setup
│   ├── identity.ts       # ACK-ID implementation
│   ├── payments.ts       # ACK-Pay implementation
│   └── types.ts          # TypeScript interfaces
├── hooks/
│   ├── useACKIdentity.ts       # Identity hook
│   ├── useACKPayment.ts        # Payment hook
│   └── useCredentialSigning.ts # EIP-712 signing
├── components/
│   └── ACKWalkthrough.tsx      # Main demo component
└── lib/arc.ts                  # Arc contract integration
```

---

## Discussion Points for Catena

### Potential Integration Paths

1. **ACK-Pay Extension**: Add optional `proofHash` field to PaymentReceiptCredential
2. **Policy Verification Service**: Standalone service ACK agents can call before payments
3. **ACK-Gate**: Generic smart contract pattern for proof-gated operations
4. **SDK Integration**: `@agentcommercekit/zkml` package for policy verification

### Open Questions

- Should policy verification be opt-in or required for certain transaction types?
- How should policy models be registered/versioned?
- What's the right abstraction for different policy types (spending, trading, etc.)?
- Should attestations live on a specific chain or be chain-agnostic?

### What We'd Love to Explore

- Feedback on the ACK-ID and ACK-Pay integration patterns used here
- Interest in collaborating on a policy verification standard
- Potential for zkML as an official ACK extension

---

## Resources

- [Agent Commerce Kit](https://agentcommercekit.com) — ACK Documentation
- [Catena Labs](https://catenalabs.com/projects/) — ACK Creator
- [ACK GitHub](https://github.com/agentcommercekit/ack) — Open Source
- [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) — zkML Prover
- [Arc Network](https://arc.network) — Testnet

---

## Contact

Interested in discussing zkML + ACK integration? We'd love to connect.

**This demo is part of the [Spending Proofs](https://github.com/hshadab/spendingproofs) project** — exploring verifiable AI agent spending policies.
