# AWS Bedrock AgentCore Gateway Integration

**zkML spending proofs** exposed as MCP-compatible tools via AWS Bedrock AgentCore Gateway.

---

## Overview

AgentCore Gateway bridges the MCP (Model Context Protocol) with REST APIs, allowing any Bedrock-hosted AI agent to generate cryptographic spending proofs.

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│    Bedrock Agent    │────▶│  AgentCore Gateway  │────▶│   Spending Prover   │
│                     │     │                     │     │                     │
│  "Generate proof    │     │  MCP → HTTP bridge  │     │  JOLT-Atlas zkML    │
│   for this spend"   │     │  SigV4 auth         │     │  ~48KB SNARK        │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## Gateway Details

| Property | Value |
|----------|-------|
| Gateway ID | `spending-proofs-czmzgtizng` |
| Region | `us-east-1` |
| Gateway URL | `https://spending-proofs-czmzgtizng.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp` |
| Auth | AWS SigV4 |

---

## MCP Tools

### generateSpendingProof

Generate a zkML proof that a spending decision follows policy.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "spending-prover-api___generateSpendingProof",
    "arguments": {
      "inputs": [0.05, 1.0, 0.2, 0.5, 0.95, 100, 5, 3600],
      "model_id": "spending-model",
      "tag": "spending"
    }
  }
}
```

**Input Array (8 values):**
| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | vendor_risk | 0-1 | Vendor risk score |
| 1 | price_ratio | 0-1 | Price relative to budget |
| 2 | category_utilization | 0-1 | Category budget utilization |
| 3 | vendor_performance | 0-1 | Historical vendor performance |
| 4 | compliance_score | 0-1 | Vendor compliance status |
| 5 | urgency_score | 0-100 | Transaction urgency |
| 6 | approval_tier | 1-5 | Required approval level |
| 7 | time_constraint | 0-86400 | Time constraint (seconds) |

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"success\":true,\"proof_hash\":\"0x1a2b3c...\",\"decision\":\"approve\",\"inference_time_ms\":4500}"
    }],
    "isError": false
  }
}
```

### getProverHealth

Check prover service health.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "spending-prover-api___getProverHealth",
    "arguments": {}
  }
}
```

---

## Usage

### AWS CLI

```bash
# List available tools
aws bedrock-agentcore invoke-gateway \
  --region us-east-1 \
  --gateway-id spending-proofs-czmzgtizng \
  --mcp-request '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Generate a proof
aws bedrock-agentcore invoke-gateway \
  --region us-east-1 \
  --gateway-id spending-proofs-czmzgtizng \
  --mcp-request '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "spending-prover-api___generateSpendingProof",
      "arguments": {
        "inputs": [0.05, 1.0, 0.2, 0.5, 0.95, 100, 5, 3600]
      }
    }
  }'
```

### Python (boto3)

```python
import boto3
import json

client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Note: As of Jan 2026, use HTTP with SigV4 signing
# See infra/agentcore/test_gateway.py for full example
```

### HTTP with SigV4

```python
import json
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

gateway_url = 'https://spending-proofs-czmzgtizng.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp'

body = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "spending-prover-api___generateSpendingProof",
        "arguments": {"inputs": [0.05, 1.0, 0.2, 0.5, 0.95, 100, 5, 3600]}
    }
}

request = AWSRequest(method='POST', url=gateway_url, data=json.dumps(body), headers={
    'Content-Type': 'application/json'
})
SigV4Auth(credentials, 'bedrock-agentcore', 'us-east-1').add_auth(request)

response = requests.post(gateway_url, data=json.dumps(body), headers=dict(request.headers))
print(response.json())
```

---

## Using with Bedrock Agents

Create a Bedrock agent that uses the gateway:

```python
# See infra/agentcore/bedrock_agent.py for full example

# Agent prompt includes:
# "You have access to a spending proof tool. When asked to verify
#  a purchase decision, use generateSpendingProof with the relevant
#  policy inputs to get a cryptographic proof."
```

---

## Running the Demo

```bash
npm run dev
# Visit /demo/agentcore
```

The demo simulates the MCP flow:
1. Connect to gateway (SigV4 auth)
2. `tools/list` - discover available tools
3. `tools/call` - generate a spending proof
4. Display proof result

---

## Performance

| Metric | Value |
|--------|-------|
| Gateway latency | <50ms |
| Proof generation | 5-15 seconds |
| Proof size | ~48KB |
| Total round-trip | 5-15 seconds |

---

## Resources

- [AgentCore Gateway Docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html)
- [MCP Protocol Spec](https://modelcontextprotocol.io/)
- [OpenAPI Spec](../../infra/agentcore/openapi.yaml)
- [Gateway Setup](../../infra/agentcore/README.md)

---

## License

MIT
