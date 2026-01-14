# Catena ACK + zkML Demo

**Extending [Catena Labs ACK](https://catenalabs.com/projects/) with zkML Policy Verification**

> [Agent Commerce Kit (ACK)](https://agentcommercekit.com) from [Catena Labs](https://catenalabs.com/projects/) provides
> the infrastructure layer for AI agent commerce: verifiable identity and payment receipts using W3C standards.
> This demo shows how **zkML extends ACK** with cryptographic proof of spending policy compliance.

## What Catena ACK Provides

ACK is an open-source framework from Catena Labs that solves two critical problems:

| ACK Component | What It Does | Standard |
|---------------|--------------|----------|
| **ACK-ID** | Verifiable agent identity | W3C DIDs |
| **ACK-Pay** | Verifiable payment receipts | W3C Verifiable Credentials |

ACK answers: **"Who is this agent?"** and **"What payments were made?"**

## What zkML Adds

zkML extends ACK with cryptographic proof of **policy compliance**:

| zkML Component | What It Does | Technology |
|----------------|--------------|------------|
| **Policy Model** | Deterministic spending policy | ONNX neural network |
| **SNARK Proof** | Proves policy was correctly evaluated | JOLT-Atlas (~48KB) |
| **Local Verification** | Gates payment on valid proof | Off-chain, <150ms |

zkML answers: **"Did the agent follow its spending rules?"**

## Together: Complete Verification

| Layer | Provider | Question Answered |
|-------|----------|-------------------|
| **Identity** | Catena ACK-ID | Who is this agent? Who owns it? |
| **Policy** | zkML (JOLT-Atlas) | Did the agent follow spending rules? |
| **Payment** | Arc Testnet | On-chain transaction record |
| **Receipt** | Catena ACK-Pay | Verifiable proof of payment |

## How zkML Extends ACK

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ACK-ID: Verifiable Agent Identity                           │
│     W3C DID + ControllerCredential signed by owner wallet       │
│     Proves: Agent identity linked to responsible party          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. zkML Proof: Policy Compliance                               │
│     JOLT-Atlas SNARK (~48KB, ~10s)                              │
│     Proves: Spending policy model executed correctly            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. ACK-Pay: Payment Receipt                                    │
│     W3C PaymentReceiptCredential signed by owner wallet         │
│     Proves: Payment made by verified agent with valid proof     │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Demo Shows

### Demo Mode vs Live Mode

| Feature | Demo Mode | Live Mode |
|---------|-----------|-----------|
| Wallet Connection | Simulated address | Real wallet (RainbowKit) |
| Identity Credential | Local generation | **EIP-712 wallet signature** |
| zkML Proof | **Real** (JOLT-Atlas) | **Real** (JOLT-Atlas) |
| USDC Payment | Mock transaction | **Real $0.01 transfer** |
| Payment Receipt | Local generation | **EIP-712 wallet signature** |

### The 5-Step Flow

1. **Identity** → Create W3C DID + sign ControllerCredential
2. **Policy** → Evaluate spending decision with ML model
3. **Proof** → Generate SNARK proof of policy execution
4. **Payment** → Execute USDC transfer on Arc Testnet
5. **Receipt** → Issue signed PaymentReceiptCredential

---

## Technical Details

### ACK-ID: Agent Identity

```typescript
interface ACKAgentIdentity {
  did: string;                          // did:key:z6Mk...
  controllerCredential: VerifiableCredential;
  ownerAddress: string;                 // 0x...
  name: string;                         // "Spending Agent"
}
```

**ControllerCredential** (EIP-712 signed in Live Mode):
- Links agent DID to owner wallet
- Proves who authorized this agent
- W3C Verifiable Credential standard

### zkML Proof

| Metric | Value |
|--------|-------|
| Proof Size | ~48KB |
| Generation Time | 4-12 seconds |
| Verification | <150ms |
| Prover | JOLT-Atlas |

### ACK-Pay: Payment Receipt

```typescript
interface ACKPaymentReceipt {
  receiptCredential: VerifiableCredential;
  txHash: string;
  amount: string;
  proofHash: string;    // Links to zkML proof
}
```

**PaymentReceiptCredential** (EIP-712 signed in Live Mode):
- Links transaction to verified proof
- Provides audit trail for compliance
- W3C Verifiable Credential standard

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

| Contract | Address |
|----------|---------|
| USDC | `0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213` |
| ProofAttestation | `0xBE9a5DF7C551324CB872584C6E5bF56799787952` |
| SpendingGateWallet | `0x6A47D13593c00359a1c5Fc6f9716926aF184d138` |

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
│   ├── useACKPayment.ts        # Payment React hook
│   └── useCredentialSigning.ts # EIP-712 signing hook
├── components/
│   ├── ACKWalkthrough.tsx      # Main demo component
│   └── ack/                    # Step components
└── app/demo/ack/
    └── page.tsx                # Demo page
```

---

## Running Live Mode

1. **Connect wallet** to Arc Testnet (Chain ID: 5042002)
2. **Get testnet USDC** from [faucet.circle.com](https://faucet.circle.com)
3. **Toggle to Live Mode** in the demo
4. **Run the demo** - you'll sign 3 things:
   - ControllerCredential (identity)
   - USDC transfer ($0.01)
   - PaymentReceiptCredential (receipt)

---

## What's Real vs Simulated

### Always Real
- zkML proof generation (JOLT-Atlas prover)
- Policy model evaluation
- W3C credential structure

### Real in Live Mode Only
- Wallet connection (RainbowKit)
- EIP-712 credential signatures
- USDC transfers on Arc Testnet

### Always Simulated
- On-chain proof attestation (not submitted)
- Agent decision-making (pre-defined)

---

## About Catena Labs ACK

[Agent Commerce Kit (ACK)](https://agentcommercekit.com) is an open-source framework from [Catena Labs](https://catenalabs.com/projects/) providing the infrastructure layer for AI agent commerce:

- **ACK-ID**: Verifiable agent identity using W3C DIDs
- **ACK-Pay**: Payment receipts using W3C Verifiable Credentials
- **ACK-Trust**: Agent reputation and trust scores

ACK provides the foundational identity and receipt infrastructure. zkML adds a complementary layer for policy verification.

### How They Work Together

| Catena ACK Provides | zkML Extends With |
|---------------------|-------------------|
| Agent identity (DID) | — |
| Owner verification (ControllerCredential) | — |
| — | Policy model execution proof |
| — | Local verification before payment |
| Payment receipts (PaymentReceiptCredential) | Proof hash linkage |

**Result**: Complete audit trail with cryptographic guarantees at every layer.

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
