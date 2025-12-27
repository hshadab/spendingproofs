# SpendingProofs Acquisition Upgrade Plan

## Goal
Transform the demo from "cool zkML demo" → "platform primitive worth buying" for Circle acquisition positioning.

---

## Phase 1: Critical Fixes (Do First - Highest ROI)

### 1.1 Fix Performance Inconsistency
**Problem:** Landing page claims ~0.7s/143ms, demo page says 4-12 seconds.

**Changes:**
- [ ] Create unified `PerformanceMetrics` component used everywhere
- [ ] Display three separate metrics:
  - Prover compute time (warm) - p50/p90
  - End-to-end latency (cold starts) - p50/p90
  - Verification cost (ms offchain, gas onchain)
- [ ] Replace "~0.7s" with actual benchmarked values
- [ ] Add link to reproducible benchmark script
- [ ] Update landing page, demo page, and Jolt-Atlas section

**Files to modify:**
- `src/app/page.tsx` (landing)
- `src/app/demo/page.tsx`
- `src/app/demo/playground/page.tsx`
- `src/components/ProofProgress.tsx`
- New: `src/components/PerformancePanel.tsx`

### 1.2 Remove Unauthorized Branding
**Problem:** Using `@arc/policy-proofs` and `prover.arc.network` without authorization.

**Changes:**
- [ ] Rename package scope: `@arc/policy-proofs` → `@icme-labs/spending-proofs`
- [ ] Change prover URL: `prover.arc.network` → `prover.spendingproofs.dev` (or similar)
- [ ] Add footer disclaimer: "Demonstration of Arc-native design. Not affiliated with Circle."
- [ ] Update all code examples on landing page

**Files to modify:**
- `src/app/page.tsx` (all code examples)
- `README.md`

---

## Phase 2: Hard Enforcement (Security Primitive)

### 2.1 Add SpendingGate Contract Demo
**Problem:** Current flow is "attestation" (logging), not "enforcement" (blocking).

**Changes:**
- [ ] Create mock `SpendingGate` contract interface
- [ ] Add enforcement demo showing:
  - Attempt spend without proof → **REVERTS**
  - Attempt spend with valid proof → **SUCCEEDS**
  - Attempt replay/modify amount → **REVERTS**
- [ ] Update payment demo flow visualization

**New components:**
- `src/components/EnforcementDemo.tsx`
- `src/lib/spendingGate.ts` (mock contract interface)

**Demo flow update:**
```
Policy Check → Proof Generation → Verification → Gated USDC Transfer
                                      ↓
                              [REVERT if invalid]
```

### 2.2 Add PolicyRegistry
**Problem:** Without registry, malicious agent can prove compliance with permissive model.

**Changes:**
- [ ] Create `PolicyRegistry` contract interface
- [ ] Define registry structure: `policyId → {modelHash, vkHash, metadataURI}`
- [ ] Add registry lookup to verification flow
- [ ] Show in UI:
  - policyId
  - modelHash
  - vkHash
  - "✅ approved policy" / "❌ unknown policy"

**New files:**
- `src/lib/policyRegistry.ts`
- `src/components/PolicyRegistryPanel.tsx`

---

## Phase 3: Transaction Intent Binding

### 3.1 Add txIntentHash to Proofs
**Problem:** Proofs aren't bound to specific transaction intent.

**Changes:**
- [ ] Define `txIntentHash` structure:
  ```typescript
  {
    chainId: number,
    usdcAddress: Address,
    sender: Address,      // agent wallet
    recipient: Address,
    amount: bigint,
    nonce: bigint,
    expiry: number,
    policyId: string,
    policyVersion: number
  }
  ```
- [ ] Update proof generation to include txIntentHash
- [ ] Update mock proof to include this commitment
- [ ] Add UI showing intent binding

### 3.2 Add Replay Attack Demo
**Changes:**
- [ ] Add button: "Replay this proof with different recipient"
- [ ] Add button: "Replay this proof with different amount"
- [ ] Show verification failure with reason

**Files to modify:**
- `src/app/demo/tamper/page.tsx`
- `src/hooks/useProofGeneration.ts`
- `src/lib/types.ts`

---

## Phase 4: Split View (Agent vs Verifier)

### 4.1 Dual-Pane Demo UI
**Problem:** Demo feels like single-party toy, not two-party protocol.

**Changes:**
- [ ] Redesign playground with split view:

**Left Pane: Agent View**
- Policy parameters (visible)
- Private context (visible)
- Generate proof button
- Full proof details

**Right Pane: Merchant/Verifier View**
- tx intent (recipient, amount, expiry)
- proof + public signals only
- policyId, modelHash, inputHash
- verification outcome + reason
- **Never sees:** policy thresholds, private signals

**New components:**
- `src/components/AgentPane.tsx`
- `src/components/VerifierPane.tsx`
- `src/components/SplitDemoLayout.tsx`

---

## Phase 5: Proof Semantics Documentation

### 5.1 Add "What is Proven" Section
**Changes:**
- [ ] Add clear table to landing page:

| Category | Field | Visibility |
|----------|-------|------------|
| **Public** | policyId | Verifier sees |
| **Public** | modelHash / vkHash | Verifier sees |
| **Public** | txIntentHash | Verifier sees |
| **Public** | decision (shouldBuy, riskScore) | Verifier sees |
| **Private** | policy thresholds/weights | Hidden |
| **Private** | private context inputs | Hidden |

**Verified Statement:**
> "Model M evaluated inputs X bound to txIntentHash T and produced decision D."

**Files to modify:**
- `src/app/page.tsx` (new section after "What the Proof Guarantees")

---

## Phase 6: Attack Demos

### 6.1 Model Substitution Attack
**Changes:**
- [ ] Demo: Agent tries permissive model
- [ ] Registry check fails
- [ ] Show: "❌ Unknown model - not in approved registry"

### 6.2 Policy Downgrade Attack
**Changes:**
- [ ] Demo: Agent uses outdated policy version
- [ ] Version check fails
- [ ] Show: "❌ Policy version mismatch - v1 required, v0 provided"

**New component:**
- `src/components/AttackDemos.tsx`
- Add to `/demo/tamper` page

---

## Phase 7: Local Reproducibility

### 7.1 Docker Compose Setup
**Changes:**
- [ ] Create `docker-compose.yml`:
  - Local chain (anvil)
  - Mock USDC contract
  - Prover service (mock)
  - Frontend
- [ ] Add `npm run demo:local`

### 7.2 E2E Test Script
**Changes:**
- [ ] Create `scripts/e2e-demo.ts`:
  1. Generate proof
  2. Verify proof
  3. Submit gated spend (success)
  4. Attempt tamper (expected fail)
  5. Attempt replay (expected fail)
- [ ] Add `npm run demo:e2e`

**New files:**
- `docker-compose.yml`
- `scripts/e2e-demo.ts`
- `Dockerfile`

---

## Phase 8: Circle Integration Paths

### 8.1 Wallet/Paymaster Gating Demo
**Changes:**
- [ ] Show: no valid proof → no sponsored execution
- [ ] Add "Paymaster Gating" demo section

### 8.2 Merchant Checkout Gating Demo
**Changes:**
- [ ] Show: merchant contract checks proof then accepts USDC
- [ ] Visual checkout flow with gating step

### 8.3 (Optional) Paywalled API Demo
**Changes:**
- [ ] Agent requests resource
- [ ] Must present proof + payment compliance
- [ ] Pays USDC, receives access

---

## Phase 9: Copy Updates

### 9.1 Lead with "Execution Control"
**Current:** "Trust Primitive for Agentic Commerce"
**New:** "Proof-Gated USDC Execution for Autonomous Agents"

### 9.2 Attestation vs Verification Honesty
Add to demo page:
> "Today: onchain attestation + offchain verification (demo)."
> "Next: full onchain SNARK verification and enforced gating."

### 9.3 (Optional) "Why Circle Buys This" Page
Private page linking:
- Arc adoption story
- Agentic payments safety
- Enterprise spend governance
- Roadmap integration

---

## Implementation Priority

### Week 1: Critical (Must Do)
1. ✅ Fix performance inconsistency
2. ✅ Remove @arc branding
3. ✅ Add enforcement demo (revert without proof)
4. ✅ Add PolicyRegistry concept

### Week 2: High Value
5. ✅ Add txIntentHash binding
6. ✅ Add replay attack demo
7. ✅ Split view (Agent vs Verifier)

### Week 3: Polish
8. ✅ Proof semantics documentation
9. ✅ Attack demos (model substitution, policy downgrade)
10. ✅ Copy updates

### Week 4: Reproducibility (Future)
11. Docker compose setup
12. E2E test script
13. Circle integration path demos

---

## Success Criteria

After these changes, a Circle exec + Arc engineer should conclude:

1. **Performance is real** - Clear metrics, reproducible benchmarks
2. **Enforcement is real** - Transactions revert without valid proofs
3. **Privacy is real** - Policy registry prevents model substitution
4. **Intent binding is real** - Replay attacks fail
5. **Integration is clear** - Maps to Circle platform surfaces
6. **Diligence is easy** - Can run locally in minutes
