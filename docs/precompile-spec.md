# Jolt-Atlas Verifier Precompile Specification

**Version**: 0.1.0-draft
**Status**: Proposal
**Target**: Arc L1

## Abstract

This document specifies a native EVM precompile for verifying Jolt-Atlas zkML proofs on Arc. The precompile enables efficient on-chain verification of zero-knowledge machine learning inference proofs, making zkML policy enforcement a first-class primitive on Arc.

## Motivation

Autonomous agents require cryptographic proof that their spending decisions followed policy. Current options for on-chain verification are:

1. **Attestation Only**: Off-chain verification, on-chain hash commit (~21k gas)
   - Pros: Cheap, simple
   - Cons: Not trustless, requires trust in verifier

2. **Solidity Verifier**: Smart contract implementing HyperKZG (~500k gas)
   - Pros: Trustless, EVM portable
   - Cons: Expensive, complex contract

3. **Native Precompile**: Protocol-level verification (~50k gas)
   - Pros: Trustless, cheap, secure
   - Cons: Requires protocol upgrade

This specification proposes option 3 as the optimal solution for Arc's agentic commerce vision.

## Specification

### Precompile Address

```
0x000000000000000000000000000000000000000f
```

Address `0x0f` is chosen as it's in the reserved precompile range and not currently allocated in the EVM specification.

### Input Format

The precompile accepts a single calldata blob with the following structure:

```
┌────────────────────────────────────────────────────────────────┐
│ Offset │ Size    │ Field          │ Description                │
├────────────────────────────────────────────────────────────────┤
│ 0      │ 32      │ policyHash     │ Keccak256 of policy VK     │
│ 32     │ 32      │ inputsHash     │ Poseidon hash of inputs    │
│ 64     │ 32      │ txIntentHash   │ Transaction binding hash   │
│ 96     │ 32      │ decisionHash   │ Hash of model outputs      │
│ 128    │ 4       │ proofLength    │ Length of proof in bytes   │
│ 132    │ var     │ proof          │ Serialized Jolt proof      │
└────────────────────────────────────────────────────────────────┘
```

### Output Format

On success, returns 32 bytes:

```
0x0000000000000000000000000000000000000000000000000000000000000001
```

On failure, returns 32 bytes with error code:

```
0x00000000000000000000000000000000000000000000000000000000000000XX
```

Where `XX` is the error code (see Error Codes section).

### Gas Schedule

| Component | Gas Cost | Notes |
|-----------|----------|-------|
| Base cost | 45,000 | Fixed overhead for precompile invocation |
| Proof data | 100/KB | Linear cost for proof deserialization |
| Point validation | 500/point | BN254 curve point validation |
| Pairing check | 3,000/pair | BN254 pairing operation |

**Typical Total**: ~50,000 gas for a standard 48KB proof

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| `0x00` | `SUCCESS` | Proof verified successfully |
| `0x01` | `INVALID_PROOF_FORMAT` | Proof bytes could not be deserialized |
| `0x02` | `INVALID_CURVE_POINT` | G1 or G2 point not on BN254 curve |
| `0x03` | `PAIRING_CHECK_FAILED` | HyperKZG pairing verification failed |
| `0x04` | `SUMCHECK_FAILED` | Sumcheck protocol verification failed |
| `0x05` | `HASH_MISMATCH` | Public inputs don't match claimed hashes |
| `0x06` | `PROOF_TOO_LARGE` | Proof exceeds maximum size (1MB) |
| `0x07` | `INVALID_VK` | Verification key format invalid |

## Proof Format

### Jolt-Atlas Proof Structure

```rust
struct JoltProof {
    // Polynomial commitments (BN254 G1 points)
    commitments: Vec<G1Affine>,

    // Opening proofs (BN254 G1 points)
    openings: Vec<G1Affine>,

    // Evaluation claims (field elements)
    evaluations: Vec<Fr>,

    // Sumcheck proof rounds
    sumcheck_rounds: Vec<SumcheckRound>,

    // Final evaluation
    final_eval: Fr,
}

struct SumcheckRound {
    // Coefficients for this round's polynomial
    coefficients: [Fr; 3],
}
```

### Serialization

Proof is serialized as:

1. **Header** (8 bytes)
   - Version (2 bytes): `0x0001`
   - Num commitments (2 bytes)
   - Num openings (2 bytes)
   - Num sumcheck rounds (2 bytes)

2. **Commitments** (64 bytes each)
   - G1 point in uncompressed form (32 bytes x, 32 bytes y)

3. **Openings** (64 bytes each)
   - G1 point in uncompressed form

4. **Evaluations** (32 bytes each)
   - BN254 scalar field element

5. **Sumcheck Rounds** (96 bytes each)
   - 3 field elements per round

6. **Final Evaluation** (32 bytes)
   - BN254 scalar field element

## Verification Algorithm

### High-Level Flow

```
1. Deserialize proof
2. Validate all curve points
3. Reconstruct public inputs from hashes
4. Verify HyperKZG commitment openings
5. Verify sumcheck protocol
6. Check final evaluation matches
7. Return success/failure
```

### HyperKZG Verification

```python
def verify_hyperkzg(commitment, point, evaluation, proof):
    # Compute pairing check
    lhs = pairing(commitment - evaluation * G1, H)
    rhs = pairing(proof, tau * H - point * H)
    return lhs == rhs
```

### Sumcheck Verification

```python
def verify_sumcheck(claim, rounds, final_eval):
    current_claim = claim
    challenges = []

    for round in rounds:
        # Verify round polynomial
        p = round.coefficients
        assert p[0] + p[1] == current_claim

        # Generate challenge via Fiat-Shamir
        challenge = hash_to_field(current_claim, p)
        challenges.append(challenge)

        # Update claim
        current_claim = evaluate_poly(p, challenge)

    # Verify final evaluation
    assert current_claim == final_eval
    return True
```

## Implementation Notes

### Rust Implementation

The precompile should be implemented in Rust for performance:

```rust
pub fn jolt_verify(input: &[u8]) -> PrecompileResult {
    // Parse input
    let (policy_hash, inputs_hash, tx_intent_hash, decision_hash, proof_bytes) =
        parse_input(input)?;

    // Deserialize proof
    let proof = JoltProof::deserialize(proof_bytes)
        .map_err(|_| PrecompileError::InvalidProofFormat)?;

    // Validate curve points
    for point in proof.all_points() {
        if !point.is_on_curve() {
            return Err(PrecompileError::InvalidCurvePoint);
        }
    }

    // Verify HyperKZG
    if !verify_hyperkzg(&proof) {
        return Err(PrecompileError::PairingCheckFailed);
    }

    // Verify sumcheck
    if !verify_sumcheck(&proof) {
        return Err(PrecompileError::SumcheckFailed);
    }

    // Verify public input hashes
    if !verify_hashes(&proof, policy_hash, inputs_hash, tx_intent_hash, decision_hash) {
        return Err(PrecompileError::HashMismatch);
    }

    Ok(PrecompileOutput::success())
}
```

### Integration with Reth

Arc uses Reth for EVM execution. The precompile integrates as:

```rust
impl Precompile for JoltVerifier {
    fn run(&self, input: &Bytes, gas_limit: u64) -> PrecompileResult {
        // Calculate gas cost
        let gas_cost = calculate_gas_cost(input.len());
        if gas_cost > gas_limit {
            return Err(PrecompileError::OutOfGas);
        }

        // Run verification
        let result = jolt_verify(input)?;

        Ok(PrecompileOutput {
            cost: gas_cost,
            output: result,
        })
    }
}
```

## Security Considerations

### Curve Point Validation

All BN254 points must be validated to be on the curve. Invalid points could lead to:
- Subgroup attacks
- Invalid pairing results
- Consensus failures between nodes

### Gas Metering

Gas costs must be carefully calibrated to prevent:
- Denial of service via expensive operations
- Underpriced operations allowing spam
- Variance in execution time across nodes

### Fiat-Shamir Security

The sumcheck uses Fiat-Shamir for non-interactivity. The hash function must:
- Be deterministic across all implementations
- Use domain separation to prevent cross-protocol attacks
- Include all relevant context in the transcript

## Testing

### Test Vectors

A comprehensive test suite should include:

1. **Valid proofs**: Known-good proofs that should verify
2. **Invalid proofs**: Proofs with each type of error
3. **Edge cases**: Maximum size proofs, minimum size proofs
4. **Gas estimation**: Verify gas costs match specification

### Fuzzing

The precompile should be fuzzed with:
- Random proof bytes
- Valid proofs with mutated bytes
- Boundary conditions for all parameters

## Deployment

### Activation

The precompile should be activated via Arc governance:

1. Deploy precompile code to all validators
2. Propose activation block height
3. Governance vote
4. Automatic activation at specified block

### Backwards Compatibility

The precompile introduces no backwards compatibility issues:
- New address, no existing code affected
- Existing contracts can optionally use the precompile
- No changes to existing EVM semantics

## References

- [Jolt: SNARKs for Virtual Machines](https://eprint.iacr.org/2023/1217)
- [HyperKZG Polynomial Commitments](https://eprint.iacr.org/2023/1217)
- [BN254 Curve Specification](https://eips.ethereum.org/EIPS/eip-196)
- [Arc Technical Documentation](https://docs.arc.network)
