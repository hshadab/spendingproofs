# OpenMind Robot Payments Demo

**Autonomous robots making USDC payments via x402 with zkML spending proof guardrails powered by [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas).**

> This demo showcases how OpenMind's OM1 robot operating system can integrate with
> zkML spending proofs to enable trustless autonomous commerce for embodied AI.

## Complete Workflow

### Step 1: Robot Encounters Payment Situation
**Tech: OpenMind OM1**

- Robot running OpenMind's OM1 operating system detects low battery (23%)
- OM1 discovers nearby ChargePoint station offering charging for $0.10 USDC via x402 protocol
- Robot needs to decide: should I pay for this service?

```
┌─────────────────────────────────────┐
│  OpenMind OM1 Robot OS              │
│  • Sensors detect low battery       │
│  • x402 service discovery           │
│  • Payment decision needed          │
└─────────────────────────────────────┘
```

---

### Step 2: LLM Contextual Reasoning (NOT Proven)
**Tech: OpenMind LLM**

- OpenMind's LLM evaluates context: "Should I charge now or keep delivering? Is this station on my route? Can I reach a cheaper station?"
- This reasoning is intelligent but **NOT cryptographically proven**
- LLM returns decision: `APPROVE` or `DENY`

```
┌─────────────────────────────────────┐
│  OpenMind LLM Decision              │
│  • Context: task priority, route    │
│  • Output: APPROVE                  │
│  • Status: Smart, NOT proven        │
└─────────────────────────────────────┘
```

---

### Step 3: Policy Model Check (IS Proven)
**Tech: ONNX Policy Model → Jolt Atlas**

- A small deterministic ONNX model checks hard spending limits:
  - Price ($0.10) ≤ max single tx ($2.00) ✓
  - Category (charging) in allowed list ✓
  - Daily spend ($3.50 + $0.10) ≤ daily limit ($10.00) ✓
  - Service reliability (98%) ≥ minimum (85%) ✓
- This model **IS what gets proven**

```
┌─────────────────────────────────────┐
│  Policy Model (spending-model.onnx) │
│  Inputs:                            │
│  • price: 0.10                      │
│  • budget: 100.00                   │
│  • spentToday: 3.50                 │
│  • dailyLimit: 10.00                │
│  • reliability: 0.98                │
│  Output: APPROVE (confidence: 0.95) │
└─────────────────────────────────────┘
```

---

### Step 4: zkML Proof Generation
**Tech: Jolt Atlas SNARK Prover**

- Jolt Atlas runs the ONNX policy model inside a SNARK circuit
- Generates cryptographic proof that:
  - The exact policy model was executed
  - With the exact inputs
  - Producing the exact output
- Proof is ~50KB, generation takes 2-5 seconds locally

```
┌─────────────────────────────────────┐
│  Jolt Atlas Prover                  │
│  • HyperKZG polynomial commitments  │
│  • BN254 curve                      │
│  • Output: 50KB SNARK proof         │
│  • proofHash: 0x7a8b3c...           │
└─────────────────────────────────────┘
```

---

### Step 5: Off-Chain Proof Verification
**Tech: Jolt Atlas Verifier (local)**

- Proof is verified locally (off-chain)
- Verification takes <150ms
- If valid: payment proceeds
- If invalid: payment blocked

```
┌─────────────────────────────────────┐
│  Local Verification                 │
│  • Verify SNARK proof               │
│  • Check: proof valid? ✓            │
│  • Time: <150ms                     │
│  • Result: AUTHORIZED               │
└─────────────────────────────────────┘
```

---

### Step 6: USDC Payment Execution
**Tech: Arc Testnet + Circle USDC**

- Robot's wallet sends $0.10 native USDC to ChargePoint
- Transaction submitted to Arc testnet
- Arc uses native USDC (18 decimals) as gas token

```
┌─────────────────────────────────────┐
│  Arc Testnet Transaction            │
│  • From: Robot wallet               │
│  • To: ChargePoint station          │
│  • Amount: 0.10 USDC                │
│  • Status: Confirmed                │
└─────────────────────────────────────┘
```

---

### Step 7: Proof Attestation (Audit Trail)
**Tech: Arc ProofAttestation Contract**

- proofHash submitted to ProofAttestation contract on Arc
- Creates immutable on-chain record for audit
- Non-blocking: payment already succeeded
- Anyone can verify: "Did this payment have a valid proof?"

```
┌─────────────────────────────────────┐
│  ProofAttestation Contract          │
│  • submitProof(proofHash, metadata) │
│  • Emits: ProofSubmitted event      │
│  • Immutable audit trail            │
└─────────────────────────────────────┘
```

---

## Tech Stack Responsibilities

| Layer | Technology | Role |
|-------|------------|------|
| **Robot OS** | OpenMind OM1 | Service discovery, wallet, x402 protocol |
| **Contextual AI** | OpenMind LLM | Smart reasoning (not proven) |
| **Policy Enforcement** | ONNX Model | Deterministic limit checks (proven) |
| **Proof Generation** | [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) | SNARK proof of policy execution |
| **Proof Verification** | Jolt Atlas | Off-chain verification (<150ms) |
| **Payment** | Arc + USDC | Native USDC transfers |
| **Audit Trail** | Arc ProofAttestation | On-chain proof attestation |

---

## The Trust Guarantee

**What's proven:** The policy model (spending limits) executed correctly.

**What's NOT proven:** The LLM's contextual reasoning.

**Why this works:** Even if the LLM makes a questionable decision ("charge now even though there's a cheaper station nearby"), the hard limits are mathematically enforced. The robot literally cannot overspend.

---

## The Problem

From OpenMind's own documentation:
> "Suitably configured system prompts will result in autonomous (and inherently unpredictable) real world actions by your robot"

When robots can autonomously spend money, owners need guarantees that spending policies are actually being followed.

## The Solution

Before every x402 USDC payment, the robot generates a zkML proof that:
- The spending decision model was executed correctly
- All policy constraints were satisfied (limits, categories, reliability)
- The inputs weren't tampered with

**No proof, no payment.**

---

## Robot Spending Policy

```typescript
interface RobotSpendingPolicy {
  dailyLimitUsdc: number;        // e.g., $10/day
  maxSingleTxUsdc: number;       // e.g., $2 per tx
  allowedCategories: string[];   // e.g., ['charging', 'navigation', 'compute']
  minServiceReliability: number; // e.g., 85%
  requireProofForAll: boolean;   // Always require zkML proof
}
```

---

## Real vs Simulated Mode

The demo includes toggles for:

| Mode | Prover | Payment | Use Case |
|------|--------|---------|----------|
| Simulated | Mock (~1.5s) | No USDC | Quick demos |
| Real | Jolt Atlas (~2-5s) | Real USDC on Arc | Full integration |

---

## Use Cases

### 1. Delivery Robots
Autonomous delivery bots paying for charging, navigation APIs, and route optimization.

### 2. Inspection Drones
Industrial inspection drones paying for cloud compute for vision processing.

### 3. Home Assistants
Home robots paying for data services, smart home APIs, and maintenance.

### 4. Digital Agents
Pure software agents paying for API access, compute, and data.

---

## OpenMind + Circle Partnership

In December 2025, OpenMind and Circle announced:
- First USDC payment infrastructure for autonomous robots
- Micropayments as small as $0.004 USDC
- High-throughput batching for thousands of tx/second
- Pilot program launching in Silicon Valley

---

## Related Resources

- [OpenMind](https://openmind.org) - Open Source AI Robot Operating System
- [x402 Integration Docs](https://docs.openmind.org/robotics/coinbase-x402)
- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) - zkML SNARK Prover
- [Circle](https://www.circle.com) - USDC Stablecoin
- [Arc Network](https://arc.network) - Testnet for this demo

---

**Part of the [Spending Proofs](https://github.com/hshadab/spendingproofs) project.**
