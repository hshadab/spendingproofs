# Skyfire Integration Configuration

## Account Setup

1. Create a Skyfire account at [app-sandbox.skyfire.xyz](https://app-sandbox.skyfire.xyz)
2. Complete identity verification (required for buyer accounts)
3. Generate an API key from the dashboard

## API Credentials

- **API Key**: `your-skyfire-api-key` (from dashboard)
- **Environment**: Sandbox or Production
- **API Base URL**: `https://api-sandbox.skyfire.xyz` (sandbox) or `https://api.skyfire.xyz` (production)
- **Dashboard**: `https://app-sandbox.skyfire.xyz` (sandbox) or `https://app.skyfire.xyz` (production)

## Seller Service (for Demo)

Use the Skyfire Official Seller service for testing:

- **Service Name**: Skyfire Official Seller - Sum of Today's date in UTC
- **Service ID**: `3b622b2f-7a2d-4ee5-86c5-58d3b8bdf73d`
- **Price**: $0.000001 per use
- **Accepted Tokens**: KYA, PAY, KYA+PAY

## API Endpoints

| Operation | Method | Endpoint |
|-----------|--------|----------|
| Create Token | POST | `/api/v1/tokens` |
| Charge Token | POST | `/api/v1/tokens/charge` |
| Introspect Token | POST | `/api/v1/tokens/introspect` |
| Get Services | GET | `/api/v1/directory/services` |

## Authentication

Header: `skyfire-api-key: <your-api-key>`

## Token Types

| Type | Description |
|------|-------------|
| `kya` | Know Your Agent - identity verification only |
| `pay` | Payment authorization only |
| `kya+pay` | Combined identity + payment token |

## Example: Create KYA+PAY Token

```bash
curl --request POST \
     --url https://api-sandbox.skyfire.xyz/api/v1/tokens \
     --header 'skyfire-api-key: <your-api-key>' \
     --header 'content-type: application/json' \
     --data '{
       "type": "kya+pay",
       "buyerTag": "zkml-demo-agent",
       "sellerServiceId": "3b622b2f-7a2d-4ee5-86c5-58d3b8bdf73d",
       "expiresAt": 1768511748,
       "tokenAmount": "0.005"
     }'
```

## Example: Use Token with Seller Service

```bash
curl --request POST \
     --url https://skyfire-official-playground-service-sandbox.skyfire.xyz \
     --header 'skyfire-pay-id: <JWT_TOKEN>' \
     --header 'Content-Type: application/json' \
     --data '{}'
```

## Environment Variables

Add to `.env.local`:

```bash
SKYFIRE_API_KEY=your-skyfire-api-key
SKYFIRE_API_URL=https://api-sandbox.skyfire.xyz
SKYFIRE_ENVIRONMENT=sandbox
NEXT_PUBLIC_SKYFIRE_ENABLED=true
SKYFIRE_DEMO_WALLET_PRIVATE_KEY=0x...  # Your wallet key for Arc transfers
```

## Integration with zkML

Our integration binds Skyfire identity to zkML proofs:

1. **KYA Token**: Proves WHO the agent is (Skyfire identity)
2. **zkML Proof**: Proves the agent followed its spending policy
3. **verificationHash**: `keccak256(proofHash, agentId, decision, confidence, timestamp)`
4. **On-Chain Attestation**: Immutable record on Arc testnet
5. **Gated Transfer**: SpendingGateWallet checks attestation before releasing funds

## Useful Links

- Skyfire Docs: https://docs.skyfire.xyz
- Sandbox Dashboard: https://app-sandbox.skyfire.xyz
- Production Dashboard: https://app.skyfire.xyz
- API Reference: https://docs.skyfire.xyz/reference/create-token
