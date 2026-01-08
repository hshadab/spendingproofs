# API Reference

This document describes the HTTP API endpoints exposed by the Spending Proofs application.

## Base URL

- **Development:** `http://localhost:3000/api`
- **Production:** `https://spendingproofs.com/api`

## Authentication

Some endpoints support optional wallet signature authentication. When enabled (`REQUIRE_SIGNATURE_AUTH=true`), requests must include:

```json
{
  "address": "0x...",
  "timestamp": 1704067200000,
  "signature": "0x..."
}
```

## Endpoints

### POST /api/prove

Generate a zkML proof for a spending decision.

**Request Body:**

```json
{
  "inputs": [0.05, 1.00, 0.20, 0.50, 0.95, 100, 5, 2.5],
  "tag": "spending",
  "address": "0x...",
  "timestamp": 1704067200000,
  "signature": "0x..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | `number[]` | Yes | Numeric spending model inputs |
| `tag` | `string` | Yes | Proof tag (usually "spending") |
| `address` | `string` | Conditional | Wallet address (if auth enabled) |
| `timestamp` | `number` | Conditional | Unix timestamp in ms (if auth enabled) |
| `signature` | `string` | Conditional | EIP-191 signature (if auth enabled) |

**Response (Success):**

```json
{
  "success": true,
  "proof": {
    "proof": "base64_encoded_proof_data",
    "proofHash": "0x...",
    "programIo": "0x...",
    "metadata": {
      "modelHash": "0x...",
      "inputHash": "0x...",
      "outputHash": "0x...",
      "proofSize": 48000,
      "generationTime": 5000,
      "proverVersion": "jolt-atlas-v0.1.0",
      "txIntentHash": "0x..."
    },
    "tag": "spending",
    "timestamp": 1704067200000
  },
  "inference": {
    "output": 1,
    "rawOutput": [1, 0.95, 0.15],
    "decision": "approve",
    "confidence": 95
  },
  "generationTimeMs": 5000
}
```

**Response (Error):**

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

**Error Codes:**

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters |
| `INVALID_SIGNATURE` | Signature verification failed |
| `SIGNATURE_EXPIRED` | Timestamp too old |
| `ADDRESS_NOT_ALLOWED` | Address not in allowlist |
| `PROVER_UNAVAILABLE` | Prover service not responding |
| `INTERNAL_ERROR` | Server-side error |

---

### GET /api/prove

Health check for the prover service.

**Response:**

```json
{
  "status": "healthy",
  "proverVersion": "jolt-atlas-v0.1.0",
  "uptime": 3600
}
```

---

### POST /api/crossmint/transfer

Execute a USDC transfer with optional proof attestation.

**Request Body:**

```json
{
  "to": "0x8ba1f109551bD432803012645Ac136ddd64DBA72",
  "amount": 10.00,
  "proofHash": "0x..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `to` | `string` | Yes | Recipient Ethereum address |
| `amount` | `number` | Yes | Amount in USDC (0.000001 - 1,000,000) |
| `proofHash` | `string` | No | zkML proof hash for audit trail |

**Response (Success):**

```json
{
  "success": true,
  "transfer": {
    "status": "success",
    "txHash": "0x...",
    "to": "0x...",
    "amount": "10.00",
    "chain": "arc-testnet",
    "proofHash": "0x...",
    "attestationTxHash": "0x...",
    "verifiedOnChain": false,
    "note": "Payment executed. Proof attested on Arc for audit trail."
  },
  "steps": [
    { "step": "Off-Chain Verification", "status": "success" },
    { "step": "USDC Transfer", "status": "success", "txHash": "0x..." },
    { "step": "Proof Attestation (Audit)", "status": "success", "txHash": "0x..." }
  ]
}
```

**Error Codes:**

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid to/amount/proofHash |
| `CONFIG_ERROR` | Server misconfiguration |
| `TRANSFER_FAILED` | Transfer execution failed |
| `INTERNAL_ERROR` | Server-side error |

---

### GET /api/crossmint/transfer

Get SpendingGate wallet information.

**Response:**

```json
{
  "success": true,
  "spendingGate": {
    "address": "0x6A47D13593c00359a1c5Fc6f9716926aF184d138",
    "balance": "1000.00",
    "dailyLimit": "1000.00",
    "maxSingleTransfer": "100.00",
    "remainingDaily": "850.00",
    "nonce": 5,
    "owner": "0x..."
  },
  "contracts": {
    "usdc": "0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213",
    "proofAttestation": "0xBE9a5DF7C551324CB872584C6E5bF56799787952",
    "spendingGate": "0x6A47D13593c00359a1c5Fc6f9716926aF184d138"
  }
}
```

---

### GET /api/crossmint/wallet

Get agent wallet information and balance.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `list` | `boolean` | If `true`, list all wallets |

**Response:**

```json
{
  "success": true,
  "wallet": {
    "address": "0x...",
    "type": "evm-mpc-wallet",
    "chain": "evm"
  },
  "balance": {
    "usdc": "100.00",
    "all": [
      { "currency": "usdc", "amount": "100.00", "chain": "arc-testnet" }
    ]
  }
}
```

---

### POST /api/crossmint/wallet

Create or retrieve agent wallet.

**Response:**

```json
{
  "success": true,
  "wallet": {
    "address": "0x...",
    "type": "evm-mpc-wallet",
    "chain": "evm",
    "createdAt": "2024-01-01T00:00:00.000Z"
  }
}
```

---

## Standard Error Response

All endpoints return errors in a consistent format:

```json
{
  "success": false,
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": ["Additional", "error", "details"]
}
```

## Rate Limiting

Currently no rate limiting is implemented. Production deployments should add rate limiting at the edge (Cloudflare, Vercel, etc.).

## CORS

The API allows CORS from configured origins. Default is same-origin only.
