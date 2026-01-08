# Security

This document describes the security model, threat analysis, and best practices for Spending Policy Proofs.

## Security Model Overview

Spending Policy Proofs provides three layers of security:

1. **Off-chain Verification** - zkML proofs verify spending decisions
2. **On-chain Attestation** - Proof hashes recorded on blockchain
3. **On-chain Enforcement** - Smart contracts enforce verified transfers

```
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYERS                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Enforcement   │ SpendingGateWallet contract          │
│                         │ - Requires attested proof             │
│                         │ - Enforces spending limits            │
│                         │ - Prevents proof replay                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Attestation   │ ProofAttestation contract             │
│                         │ - Records proof hashes on-chain       │
│                         │ - Immutable audit trail               │
│                         │ - Public verifiability                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Verification  │ Jolt-Atlas zkML Prover               │
│                         │ - SNARK proof generation              │
│                         │ - Cryptographic binding               │
│                         │ - Deterministic outputs               │
└─────────────────────────────────────────────────────────────────┘
```

## Cryptographic Security

### Proof Binding

Proofs are cryptographically bound to transaction intents:

```typescript
txIntentHash = keccak256(encodePacked(
  chainId,       // Prevent cross-chain replay
  amount,        // Exact transfer amount
  recipient,     // Specific recipient
  nonce,         // Unique per-transaction
  expiry,        // Time-bound validity
  policyId       // Policy version tracking
))
```

### Input Hash Verification

All proof inputs are hashed and included in the proof:

- `modelHash` - Neural network model checksum
- `inputHash` - Input vector hash
- `outputHash` - Output vector hash

This enables:
- Tamper detection
- Reproducibility verification
- Audit trail

### Secure Random Generation

All random values use `crypto.getRandomValues()`:

```typescript
import { generateSecureBytes32, generateSecureId } from '@/lib/crypto';

// Generates cryptographically secure 32-byte hash
const proofId = generateSecureBytes32();

// Generates secure ID with prefix
const requestId = generateSecureId('req');
```

## Replay Protection

### On-chain Replay Prevention

1. **Nonce tracking** - Each proof includes a unique nonce
2. **Used proof registry** - SpendingGateWallet tracks used proofs
3. **Expiry timestamps** - Proofs expire after configurable duration

```solidity
// SpendingGateWallet.sol
mapping(bytes32 => bool) public usedProofs;

function gatedTransfer(..., bytes32 proofHash, ...) external {
    require(!usedProofs[proofHash], "ProofAlreadyUsed");
    usedProofs[proofHash] = true;
    // ... execute transfer
}
```

### Off-chain Replay Prevention

1. **Timestamp verification** - Requests must be recent
2. **Signature binding** - Requests signed by wallet
3. **Cache invalidation** - Proofs can be invalidated on policy change

## Authentication

### Signature Authentication

Optional wallet signature authentication for API requests:

```typescript
// Enable via environment
REQUIRE_SIGNATURE_AUTH=true
ALLOWED_PROVER_ADDRESSES=0x123...,0x456...
```

**Message Format:**
```
Spending Proofs Authentication

Action: Generate proof
Tag: spending
Input Hash: 0x...
Timestamp: 1704067200000
```

**Verification:**
- Signature validity (EIP-191)
- Timestamp within window (5 minute expiry)
- Clock skew tolerance (10 seconds)
- Optional address allowlist

### API Security

- No sensitive data in URLs
- Request body validation
- Structured error responses (no stack traces)
- Rate limiting (recommended at edge)

## Input Validation

All inputs are validated before processing:

```typescript
import { validateTransferInput, validateAddress } from '@/lib/validation';

// Validates: to (address), amount (positive number), proofHash (bytes32)
const result = validateTransferInput({ to, amount, proofHash });
if (!result.valid) {
  return errorResponse(result.errors);
}
```

### Validation Rules

| Field | Validation |
|-------|------------|
| Addresses | Valid Ethereum address (checksum) |
| Amounts | Positive number, within bounds |
| Proof hashes | 0x + 64 hex characters |
| Timestamps | Within clock skew tolerance |
| Strings | Length limits, no injection |

## Secret Management

### Environment Variables

Sensitive configuration via environment:

```bash
# Server-side only (never exposed to client)
CROSSMINT_SERVER_KEY=sk_staging_...
DEMO_WALLET_PRIVATE_KEY=0x...

# Client-safe (NEXT_PUBLIC_ prefix)
NEXT_PUBLIC_ARC_RPC=https://rpc.testnet.arc.network
```

### Best Practices

1. **Never commit secrets** - Use `.env.local` (in `.gitignore`)
2. **Rotate regularly** - Especially after any potential exposure
3. **Principle of least privilege** - Only required permissions
4. **Audit access** - Log secret usage

### Testnet vs Production

| Environment | Approach |
|-------------|----------|
| Testnet | Test keys in `.env.local`, acceptable |
| Production | Use secret manager (AWS Secrets, Vault) |

## Smart Contract Security

### Access Control

```solidity
// SpendingGateWallet.sol
modifier onlyOwner() {
    require(msg.sender == owner, "NotOwner");
    _;
}

function updateLimits(...) external onlyOwner { ... }
function emergencyWithdraw(...) external onlyOwner { ... }
```

### Spending Limits

- `dailyLimit` - Maximum daily spending
- `maxSingleTransfer` - Maximum per-transaction
- `remainingDailyAllowance` - Tracks daily usage

### Emergency Controls

- `emergencyWithdraw` - Owner can recover funds
- `updateLimits` - Owner can adjust limits
- Pause functionality (recommended for production)

## Threat Model

### Threats Addressed

| Threat | Mitigation |
|--------|------------|
| Proof forgery | SNARK cryptographic binding |
| Replay attacks | Nonce + used proof tracking |
| Unauthorized access | Signature authentication |
| Input manipulation | Hash verification |
| Clock manipulation | Clock skew tolerance (10s) |
| Front-running | TxIntent binding |

### Out of Scope

| Threat | Notes |
|--------|-------|
| Prover compromise | Trust assumption for alpha |
| Private key theft | User responsibility |
| 51% attacks | Chain security assumption |
| DoS attacks | Implement rate limiting |

## Incident Response

### If You Discover a Vulnerability

1. **Do not** disclose publicly
2. Email security@example.com with details
3. Include reproduction steps
4. Allow 90 days for patch

### Emergency Contacts

- Security: security@example.com
- Contract admin: admin@example.com

## Audit Status

| Component | Audit Status |
|-----------|--------------|
| Smart Contracts | Not audited (testnet) |
| SDK | Internal review |
| Prover | Jolt library audited |

**Note:** This is testnet alpha software. Do not use with real funds until audited.

## Security Checklist

### For Developers

- [ ] Use `createLogger` instead of `console.log`
- [ ] Validate all inputs with `@/lib/validation`
- [ ] Use `generateSecureBytes32` for random values
- [ ] Never expose private keys or API secrets
- [ ] Follow error response format (no stack traces)
- [ ] Add tests for security-critical code

### For Deployers

- [ ] Rotate all secrets before production
- [ ] Enable signature authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Restrict contract admin access
- [ ] Review environment variables

### For Users

- [ ] Verify contract addresses
- [ ] Check transaction details before signing
- [ ] Use hardware wallets for significant funds
- [ ] Monitor wallet activity
