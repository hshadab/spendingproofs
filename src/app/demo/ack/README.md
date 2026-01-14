# Catena ACK + zkML Demo

**Extending Catena ACK with zkML Spending Policy Verification**

> **Catena ACK** provides the identity and receipt infrastructure for agentic commerce — verifiable agent identity (ACK-ID) and payment receipts (ACK-Pay) using W3C standards.
> **zkML extends ACK** with cryptographic proof that spending policies were correctly evaluated.

## What Catena ACK Provides

[Agent Commerce Kit (ACK)](https://agentcommercekit.com) from [Catena Labs](https://catenalabs.com/projects/) is an open-source framework for AI agent commerce:

| ACK Component | What It Does | Standard |
|---------------|--------------|----------|
| **ACK-ID** | Verifiable agent identity | W3C DIDs |
| **ACK-Pay** | Verifiable payment receipts | W3C Verifiable Credentials |

ACK verifies: **"Who is this agent?"** and **"What payments were made?"**

## What zkML Adds

zkML extends ACK with cryptographic proof of **spending policy compliance**.

### What is a Spending Policy?

The ML spending policy checks if a purchase should be approved by evaluating multiple factors together:

| Factor | What It Checks | Example |
|--------|----------------|---------|
| **Purchase Amount** | Is this within budget? | $0.01 vs $100 budget |
| **Vendor Risk Score** | Is this vendor trustworthy? | 15% risk (low) |
| **Vendor Track Record** | How has this vendor performed? | 92% (excellent) |
| **Category Budget** | Is there category allocation left? | $15 remaining |
| **Compliance Status** | Does vendor meet requirements? | Verified |

The spending policy evaluates all these factors together and outputs APPROVE or REJECT with a confidence score.

### Why Prove It?

zkML generates a cryptographic proof (SNARK) that the spending policy model was executed correctly — proving the agent checked vendor risk, budget, and compliance before approving a purchase.

| zkML Component | What It Does |
|----------------|--------------|
| **Spending Policy Model** | ONNX neural network evaluating 6 factors |
| **zkML Policy Proof** | Cryptographic proof policy ran correctly (~48KB) |
| **On-Chain Attestation** | Proof verification result recorded on Arc |
| **Gated USDC Transfer** | SpendingGateWallet releases funds only if attested |

## Together: Complete Verification Stack

| Layer | Provider | What It Verifies |
|-------|----------|------------------|
| **Identity** | ACK-ID | Who is this agent? Who owns it? |
| **Spending Policy** | zkML | Did it check vendor risk, budget, and compliance? |
| **Attestation** | Arc | Is verification recorded on-chain? |
| **Payment** | SpendingGateWallet | Was attestation verified before funds released? |
| **Receipt** | ACK-Pay | Auditable proof of payment |

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ACK-ID: Verifiable Agent Identity                           │
│     W3C DID + ControllerCredential signed by owner wallet       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. zkML: Spending Policy Verification                          │
│     ML policy checks vendor risk, budget, compliance            │
│     JOLT-Atlas SNARK proves policy executed correctly           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. On-Chain: Proof Verification Attestation                    │
│     verificationHash attested to ProofAttestation contract      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Gated USDC Transfer + USDC Payment Receipt                  │
│     SpendingGateWallet verifies attestation, releases funds     │
│     PaymentReceiptCredential issued with proof linkage          │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Demo Shows

### Demo Mode vs Live Mode

| Feature | Demo Mode | Live Mode |
|---------|-----------|-----------|
| Wallet Connection | Simulated address | Real wallet (RainbowKit) |
| Identity Credential | Local generation | **EIP-712 wallet signature** |
| zkML Policy Proof | **Real** (JOLT-Atlas) | **Real** (JOLT-Atlas) |
| Proof Verification Attestation | Mock tx hash | **Real on-chain tx** |
| Gated USDC Transfer | Mock tx hash | **Real SpendingGateWallet tx** |
| USDC Payment Receipt | Local generation | **EIP-712 wallet signature** |

### The 6-Phase Flow

1. **Introduction** → Overview of ACK + zkML integration
2. **ACK-ID** → Verifiable agent identity established
3. **zkML Policy Proof** → Spending policy cryptographically verified
4. **Payment** → Proof verification attested, gated USDC transfer executed
5. **Receipt** → USDC payment receipt issued
6. **Complete** → Full audit trail summary

---

## Technical Details

### Spending Policy Model

The ONNX spending policy model evaluates 6 inputs:

```typescript
interface SpendingPolicyInput {
  purchaseAmount: number;      // Amount to spend
  availableBudget: number;     // Total budget
  vendorRiskScore: number;     // 0-1 (lower = safer)
  vendorTrackRecord: number;   // 0-1 (higher = better)
  categoryBudgetLeft: number;  // Category allocation
  complianceStatus: boolean;   // Vendor compliance
}

// Output
interface SpendingPolicyDecision {
  approved: boolean;           // APPROVE or REJECT
  confidence: number;          // 0-1 confidence score
}
```

### zkML Policy Proof

| Metric | Value |
|--------|-------|
| Proof Size | ~48KB |
| Generation Time | 4-12 seconds |
| Verification | <150ms |
| Prover | JOLT-Atlas |

### ACK-ID: Agent Identity

```typescript
interface ACKAgentIdentity {
  did: string;                          // did:key:z6Mk...
  controllerCredential: VerifiableCredential;
  ownerAddress: string;                 // 0x...
  name: string;                         // "Spending Agent"
}
```

### ACK-Pay: Payment Receipt

```typescript
interface ACKPaymentReceipt {
  receiptCredential: VerifiableCredential;
  txHash: string;
  attestationTxHash: string;
  amount: string;
  proofHash: string;
}
```

---

## Arc Testnet

| Property | Value |
|----------|-------|
| Chain ID | 5042002 |
| RPC | `https://rpc.testnet.arc.network` |
| Explorer | [testnet.arcscan.app](https://testnet.arcscan.app) |
| Gas Token | USDC |
| Faucet | [faucet.circle.com](https://faucet.circle.com) |

### Contracts

| Contract | Address | Purpose |
|----------|---------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` | Token transfers |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` | Store verification attestations |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` | Gated transfers |

---

## Key Files

```
src/
├── lib/ack/
│   ├── client.ts         # ACK SDK initialization
│   ├── identity.ts       # ACK-ID implementation
│   ├── payments.ts       # ACK-Pay implementation
│   └── types.ts          # TypeScript interfaces
├── hooks/
│   ├── useACKIdentity.ts       # Identity React hook
│   ├── useACKPayment.ts        # Payment React hook (gated transfer)
│   └── useCredentialSigning.ts # EIP-712 signing hook
├── components/
│   └── ACKWalkthrough.tsx      # Main demo component
├── app/
│   ├── demo/ack/page.tsx       # Demo page
│   └── api/ack/transfer/       # Gated transfer API
└── lib/arc.ts                  # Arc contract integration
```

---

## Running Live Mode

1. **Connect wallet** to Arc Testnet (Chain ID: 5042002)
2. **Get testnet USDC** from [faucet.circle.com](https://faucet.circle.com)
3. **Toggle to Live Mode** in the demo
4. **Run the demo**:
   - ControllerCredential signed by your wallet
   - zkML policy proof generated (real JOLT-Atlas)
   - Proof verification attested on-chain (real tx)
   - Gated USDC transfer via SpendingGateWallet (real tx)
   - USDC PaymentReceiptCredential signed by your wallet

---

## What's Real vs Simulated

### Always Real
- zkML policy proof generation (JOLT-Atlas prover)
- Spending policy model evaluation
- W3C credential structure

### Real in Live Mode
- Wallet connection (RainbowKit)
- EIP-712 credential signatures
- **On-chain proof verification attestation** (ProofAttestation contract)
- **Gated USDC transfer** (SpendingGateWallet contract)

### Always Simulated
- Agent decision-making (pre-defined policy inputs)

---

## About Catena Labs ACK

[Agent Commerce Kit (ACK)](https://agentcommercekit.com) is an open-source framework from [Catena Labs](https://catenalabs.com/projects/) providing the infrastructure layer for AI agent commerce:

- **ACK-ID**: Verifiable agent identity using W3C DIDs
- **ACK-Pay**: Payment receipts using W3C Verifiable Credentials

ACK provides the identity and receipt infrastructure. zkML extends ACK with spending policy verification.

### How They Work Together

| ACK Provides | zkML Extends With |
|--------------|-------------------|
| Agent identity (DID) | — |
| Owner verification | — |
| — | Spending policy verification |
| — | On-chain attestation |
| — | SpendingGateWallet enforcement |
| Payment receipts | Proof + attestation linkage |

**Together**: Verifiable identity + spending policy verification + auditable receipts.

---

## Related Resources

- [Catena Labs](https://catenalabs.com/projects/) - Creator of Agent Commerce Kit
- [Agent Commerce Kit](https://agentcommercekit.com) - ACK documentation
- [ACK GitHub](https://github.com/agentcommercekit/ack) - Open source repository
- [JOLT-Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML prover
- [Arc Network](https://arc.network) - Testnet
- [Circle USDC Faucet](https://faucet.circle.com) - Get testnet tokens

---

**Part of the [Spending Proofs](https://github.com/hshadab/spendingproofs) project.**
