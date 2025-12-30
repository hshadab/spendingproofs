# @icme-labs/spending-proofs-cli

CLI for generating and verifying zkML spending proofs.

## Installation

```bash
npm install -g @icme-labs/spending-proofs-cli
```

Or use directly with npx:

```bash
npx @icme-labs/spending-proofs-cli prove --help
```

## Commands

### prove

Generate a SNARK proof for a spending decision.

```bash
arc-prove prove \
  --price 0.05 \
  --budget 1.0 \
  --spent 0.2 \
  --limit 0.5

# Output:
# 🔐 Generating zkML proof...
#
# Decision: ✅ APPROVE
# Confidence: 92%
# Risk Score: 8%
#
# Proof generated in 6.2s
# Proof hash: 0x7a8b...3c4d
# Proof size: 48.5 KB
```

**Options:**

| Option | Description | Required |
|--------|-------------|----------|
| `--price <usdc>` | Purchase price in USDC | Yes |
| `--budget <usdc>` | Available budget in USDC | Yes |
| `--spent <usdc>` | Amount spent today in USDC | Yes |
| `--limit <usdc>` | Daily spending limit in USDC | Yes |
| `--success-rate <rate>` | Service success rate (0-1) | No (default: 0.95) |
| `--total-calls <n>` | Total service calls | No (default: 100) |
| `--category-purchases <n>` | Purchases in category | No (default: 5) |
| `--hours-since-last <h>` | Hours since last purchase | No (default: 2) |
| `--prover <url>` | Prover service URL | No (default: http://localhost:3001) |
| `--json` | Output as JSON | No |

### decide

Run spending decision locally without generating a proof. Useful for quick policy evaluation.

```bash
arc-prove decide \
  --price 0.05 \
  --budget 1.0 \
  --spent 0.2 \
  --limit 0.5

# Output:
# Decision: ✅ APPROVE
# Confidence: 100%
# Risk Score: 0%
```

### health

Check if the prover service is running and healthy.

```bash
arc-prove health

# Output:
# ✅ Prover is healthy
#    Models: spending-model
```

### check-attestation

Check if a proof hash is attested on Arc chain.

```bash
arc-prove check-attestation 0x7a8b...3c4d

# Output:
# ✅ Proof is attested on-chain
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--rpc <url>` | Arc RPC URL | https://rpc.testnet.arc.network |
| `--contract <address>` | ProofAttestation contract | 0xBE9a5DF7C551324CB872584C6E5bF56799787952 |

## JSON Output

Use `--json` for machine-readable output:

```bash
arc-prove prove --price 0.05 --budget 1.0 --spent 0.2 --limit 0.5 --json
```

```json
{
  "success": true,
  "decision": {
    "shouldBuy": true,
    "confidence": 0.92,
    "riskScore": 0.08
  },
  "proof": {
    "hash": "0x7a8b...3c4d",
    "size": 49664,
    "generationTime": 6.2
  }
}
```

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Generate spending proof
  run: |
    npx @icme-labs/spending-proofs-cli prove \
      --price ${{ inputs.price }} \
      --budget ${{ inputs.budget }} \
      --spent ${{ inputs.spent }} \
      --limit ${{ inputs.limit }} \
      --prover ${{ secrets.PROVER_URL }} \
      --json > proof.json
```

## Requirements

- Node.js 18+
- Prover service running (for `prove` command)

## License

MIT
