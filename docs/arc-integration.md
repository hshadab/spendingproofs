# Arc Integration Guide

This guide explains how Spending Proofs integrates with Arc as a first-class primitive through a native Jolt-Atlas verifier precompile.

## Overview

Spending Proofs is designed to be more than an application on Arc—it's infrastructure that makes Arc the definitive chain for agentic commerce. The integration path moves from simple attestation to full protocol-level verification.

## Integration Phases

### Phase 1: Proof Attestation (Current)

Today, Spending Proofs works by:

1. Agent generates spending proof off-chain via Jolt-Atlas
2. Proof is verified by the SDK
3. Proof hash is committed to Arc as an attestation
4. Merchants can verify the attestation before accepting payment

```typescript
import { SpendingProofs } from '@icme-labs/spending-proofs';

// Generate proof
const proof = await SpendingProofs.prove({
  priceUsdc: 0.05,
  budgetUsdc: 1.00,
  spentTodayUsdc: 0.20,
  dailyLimitUsdc: 0.50,
  serviceSuccessRate: 0.95,
  serviceTotalCalls: 100,
  purchasesInCategory: 5,
  timeSinceLastPurchase: 2
});

// Commit attestation to Arc
const txHash = await SpendingProofs.attest(proof);
```

**Gas Cost**: ~21,000 (simple storage write)

### Phase 2: Solidity Verifier (Next)

Full on-chain verification via a Solidity smart contract that implements the HyperKZG verifier:

```solidity
contract SpendingProofVerifier {
    function verify(
        bytes calldata proof,
        bytes32 policyHash,
        bytes32 inputsHash,
        bytes32 txIntentHash
    ) external view returns (bool) {
        // HyperKZG verification using ecAdd, ecMul, ecPairing precompiles
        // ... ~500k gas
    }
}
```

**Gas Cost**: ~500,000 (Solidity-based verification)

### Phase 3: Native Precompile (Vision)

Arc adds a native precompile specifically optimized for Jolt-Atlas proof verification:

```solidity
address constant JOLT_VERIFIER = address(0x0f);

function verifySpendingProof(
    bytes calldata proof,
    bytes32 policyHash,
    bytes32 inputsHash,
    bytes32 txIntentHash
) external view returns (bool) {
    (bool success, bytes memory result) = JOLT_VERIFIER.staticcall(
        abi.encode(proof, policyHash, inputsHash, txIntentHash)
    );
    return success && abi.decode(result, (bool));
}
```

**Gas Cost**: ~50,000 (native implementation)

## Why Native Precompile?

### 10x Gas Efficiency

| Verification Method | Gas Cost | Cost at $0.001/gas |
|---------------------|----------|-------------------|
| Solidity Verifier | ~500,000 | $0.50 |
| Native Precompile | ~50,000 | $0.05 |

For agents making hundreds of transactions per day, this difference is significant.

### Protocol-Level Security

A native precompile means:

- Verification happens at the consensus level
- No smart contract vulnerabilities
- Consistent, audited implementation
- Upgradeable via Arc governance

### Arc-Exclusive Advantage

This creates a moat for Arc in agentic commerce:

- Other chains: 500k gas for Solidity verification
- Arc: 50k gas for native verification
- Agents naturally gravitate to Arc for cost efficiency

## Technical Architecture

### Precompile Implementation

The native precompile handles:

1. **Proof Deserialization**: Parse the ~48KB Jolt proof
2. **Point Validation**: Verify BN254 curve points are valid
3. **HyperKZG Verification**: Execute polynomial commitment checks
4. **Sumcheck Protocol**: Verify lookup-based computation

### Integration with Malachite

The precompile integrates with Arc's Malachite consensus:

```
┌─────────────────────────────────────────────┐
│              Malachite Consensus             │
├─────────────────────────────────────────────┤
│                                             │
│  Transaction with spending proof            │
│         │                                   │
│         ▼                                   │
│  ┌─────────────────────────────┐            │
│  │  EVM Execution              │            │
│  │                             │            │
│  │  STATICCALL to 0x0f ───────►│────┐       │
│  │                             │    │       │
│  └─────────────────────────────┘    │       │
│                                     ▼       │
│  ┌─────────────────────────────────────┐    │
│  │  Jolt-Atlas Verifier Precompile     │    │
│  │                                     │    │
│  │  - HyperKZG verification            │    │
│  │  - Sumcheck protocol                │    │
│  │  - BN254 pairing operations         │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
│         │                                   │
│         ▼                                   │
│  Return: true/false                         │
│                                             │
└─────────────────────────────────────────────┘
```

## Smart Contract Integration

### SpendingGate Contract

A reference implementation for gating USDC transfers:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract SpendingGate {
    address constant JOLT_VERIFIER = address(0x0f);
    IERC20 public immutable usdc;

    mapping(bytes32 => bool) public usedProofs;

    event SpendingApproved(
        address indexed agent,
        address indexed recipient,
        uint256 amount,
        bytes32 proofHash
    );

    constructor(address _usdc) {
        usdc = IERC20(_usdc);
    }

    function executeSpending(
        address recipient,
        uint256 amount,
        bytes calldata proof,
        bytes32 policyHash,
        bytes32 inputsHash,
        bytes32 txIntentHash
    ) external {
        // Prevent proof reuse
        bytes32 proofHash = keccak256(proof);
        require(!usedProofs[proofHash], "Proof already used");
        usedProofs[proofHash] = true;

        // Verify spending proof via native precompile
        (bool success, bytes memory result) = JOLT_VERIFIER.staticcall(
            abi.encode(proof, policyHash, inputsHash, txIntentHash)
        );
        require(success && abi.decode(result, (bool)), "Invalid proof");

        // Verify txIntentHash matches transaction
        bytes32 expectedHash = keccak256(abi.encode(
            msg.sender,
            recipient,
            amount,
            block.timestamp
        ));
        require(txIntentHash == expectedHash, "Intent mismatch");

        // Execute transfer
        require(usdc.transferFrom(msg.sender, recipient, amount), "Transfer failed");

        emit SpendingApproved(msg.sender, recipient, amount, proofHash);
    }
}
```

### Agent Wallet Integration

For smart wallets with embedded agents:

```solidity
contract AgentWallet {
    address constant JOLT_VERIFIER = address(0x0f);
    bytes32 public policyHash;

    modifier requiresSpendingProof(
        bytes calldata proof,
        bytes32 inputsHash,
        bytes32 txIntentHash
    ) {
        (bool success, bytes memory result) = JOLT_VERIFIER.staticcall(
            abi.encode(proof, policyHash, inputsHash, txIntentHash)
        );
        require(success && abi.decode(result, (bool)), "Invalid spending proof");
        _;
    }

    function agentSpend(
        address to,
        uint256 amount,
        bytes calldata proof,
        bytes32 inputsHash,
        bytes32 txIntentHash
    ) external requiresSpendingProof(proof, inputsHash, txIntentHash) {
        // Execute spending
    }
}
```

## SDK Usage

### TypeScript SDK

```typescript
import { SpendingProofs, ArcProvider } from '@icme-labs/spending-proofs';

// Initialize with Arc provider
const provider = new ArcProvider('https://rpc.arc.network');
const client = new SpendingProofs({ provider });

// Generate and submit in one call
const result = await client.proveAndExecute({
  inputs: {
    priceUsdc: 0.05,
    budgetUsdc: 1.00,
    // ... other inputs
  },
  recipient: '0x...',
  amount: 50000n, // 0.05 USDC in 6 decimals
});

console.log('Transaction:', result.txHash);
console.log('Proof verified on-chain:', result.verified);
```

### Verification

```typescript
// Verify an existing proof
const isValid = await client.verify({
  proof: proofBytes,
  policyHash: '0x...',
  inputsHash: '0x...',
  txIntentHash: '0x...'
});
```

## Migration Path

### From Attestation to Native Verification

When Arc deploys the native precompile, migration is straightforward:

1. **No SDK Changes**: The SDK automatically detects native precompile availability
2. **Contract Updates**: Deploy new contracts that use the precompile
3. **Backwards Compatible**: Old attestation-based flows continue to work

```typescript
// SDK automatically uses best available verification method
const result = await client.proveAndExecute({
  // ... same API
});

// Check which method was used
console.log('Verification method:', result.verificationMethod);
// Output: 'native-precompile' | 'solidity-verifier' | 'attestation'
```

## Security Considerations

### Proof Binding

Every proof is bound to a specific transaction intent via `txIntentHash`:

```
txIntentHash = keccak256(
  sender,
  recipient,
  amount,
  nonce,
  expiry
)
```

This prevents:
- Proof reuse across transactions
- Proof replay attacks
- Amount/recipient manipulation

### Policy Verification

Merchants can verify an agent's policy is acceptable:

```typescript
// Merchant's acceptable policies
const ACCEPTED_POLICIES = new Set([
  '0x123...', // Conservative policy
  '0x456...', // Standard policy
]);

// Verify before accepting payment
const policyHash = await client.getPolicyHash(proof);
if (!ACCEPTED_POLICIES.has(policyHash)) {
  throw new Error('Unacceptable agent policy');
}
```

## Next Steps

1. **Testnet Deployment**: Test the Solidity verifier on Arc testnet
2. **Precompile Proposal**: Submit AIP (Arc Improvement Proposal) for native precompile
3. **SDK Release**: Publish TypeScript SDK with full Arc integration
4. **Ecosystem Adoption**: Partner with wallets and protocols for integration
