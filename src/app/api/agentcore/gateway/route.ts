'use server';

import { NextRequest, NextResponse } from 'next/server';
import { SignatureV4 } from '@aws-sdk/signature-v4';
import { Sha256 } from '@aws-crypto/sha256-js';
import { HttpRequest } from '@aws-sdk/protocol-http';

// Gateway configuration
const GATEWAY_CONFIG = {
  gatewayId: 'spending-proofs-czmzgtizng',
  region: 'us-east-1',
  endpoint: 'https://spending-proofs-czmzgtizng.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp',
};

interface McpRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface GatewayRequestBody {
  method: 'tools/list' | 'tools/call';
  toolName?: string;
  arguments?: Record<string, unknown>;
}

/**
 * POST /api/agentcore/gateway
 *
 * Makes real MCP calls to AWS AgentCore Gateway with SigV4 signing.
 *
 * Request body:
 * - method: 'tools/list' | 'tools/call'
 * - toolName: (for tools/call) the tool to invoke
 * - arguments: (for tools/call) tool arguments
 */
export async function POST(request: NextRequest) {
  try {
    const body: GatewayRequestBody = await request.json();

    // Check for AWS credentials
    const accessKeyId = process.env.AWS_ACCESS_KEY_ID;
    const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
    const sessionToken = process.env.AWS_SESSION_TOKEN;

    if (!accessKeyId || !secretAccessKey) {
      return NextResponse.json({
        success: false,
        error: 'AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.',
        simulated: true,
        response: getSimulatedResponse(body),
      });
    }

    // Build MCP request
    const mcpRequest: McpRequest = {
      jsonrpc: '2.0',
      id: Date.now(),
      method: body.method,
    };

    if (body.method === 'tools/call' && body.toolName) {
      mcpRequest.params = {
        name: body.toolName,
        arguments: body.arguments || {},
      };
    }

    // Create HTTP request for signing
    const url = new URL(GATEWAY_CONFIG.endpoint);
    const httpRequest = new HttpRequest({
      method: 'POST',
      hostname: url.hostname,
      path: url.pathname,
      headers: {
        'Content-Type': 'application/json',
        'Host': url.hostname,
      },
      body: JSON.stringify(mcpRequest),
    });

    // Sign with SigV4
    const signer = new SignatureV4({
      credentials: {
        accessKeyId,
        secretAccessKey,
        sessionToken,
      },
      region: GATEWAY_CONFIG.region,
      service: 'bedrock-agentcore',
      sha256: Sha256,
    });

    const signedRequest = await signer.sign(httpRequest);

    // Make the actual request with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    try {
      const response = await fetch(GATEWAY_CONFIG.endpoint, {
        method: 'POST',
        headers: signedRequest.headers as Record<string, string>,
        body: JSON.stringify(mcpRequest),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        return NextResponse.json({
          success: false,
          error: `Gateway error: ${response.status} ${response.statusText}`,
          details: errorText,
          simulated: false,
        }, { status: response.status });
      }

      const mcpResponse = await response.json();

      return NextResponse.json({
        success: true,
        simulated: false,
        request: mcpRequest,
        response: mcpResponse,
      });
    } catch (fetchError) {
      // Timeout or network error - fall back to simulation
      clearTimeout(timeoutId);
      console.log('Gateway timeout/error, using simulation:', fetchError);
      return NextResponse.json({
        success: true,
        simulated: true,
        request: mcpRequest,
        response: getSimulatedResponse(body),
        note: 'Real gateway timed out, using simulated response',
      });
    }

  } catch (error) {
    console.error('AgentCore gateway error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      simulated: true,
      response: getSimulatedResponse({ method: 'tools/list' }),
    }, { status: 200 }); // Return 200 with simulated response instead of 500
  }
}

/**
 * GET /api/agentcore/gateway
 *
 * Returns gateway configuration and status.
 */
export async function GET() {
  const hasCredentials = !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY);

  return NextResponse.json({
    gateway: {
      id: GATEWAY_CONFIG.gatewayId,
      region: GATEWAY_CONFIG.region,
      endpoint: GATEWAY_CONFIG.endpoint,
    },
    configured: hasCredentials,
    tools: [
      { name: 'spending-prover-api___generateSpendingProof', description: 'Generate zkML spending proof' },
      { name: 'spending-prover-api___getProverHealth', description: 'Health check for prover service' },
    ],
  });
}

/**
 * Simulated responses when AWS credentials are not available.
 */
function getSimulatedResponse(body: GatewayRequestBody) {
  if (body.method === 'tools/list') {
    return {
      jsonrpc: '2.0',
      id: Date.now(),
      result: {
        tools: [
          {
            name: 'spending-prover-api___generateSpendingProof',
            description: 'Generate zkML spending proof',
            inputSchema: {
              type: 'object',
              properties: {
                inputs: { type: 'array', items: { type: 'number' } },
                model_id: { type: 'string' },
                tag: { type: 'string' },
              },
              required: ['inputs'],
            },
          },
          {
            name: 'spending-prover-api___getProverHealth',
            description: 'Health check',
            inputSchema: { type: 'object', properties: {} },
          },
        ],
      },
    };
  }

  if (body.method === 'tools/call') {
    // For generateSpendingProof, return simulated proof data
    if (body.toolName?.includes('generateSpendingProof')) {
      const proofHash = '0x' + Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('');
      return {
        jsonrpc: '2.0',
        id: Date.now(),
        result: {
          isError: false,
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              proof_hash: proofHash,
              decision: 'approve',
              confidence: 0.96,
              model_id: 'spending-model',
              inference_time_ms: 4500,
              proof_size_bytes: 48000,
              risk_score: 0.04,
              _simulated: true,
            }),
          }],
        },
      };
    }

    // For health check
    return {
      jsonrpc: '2.0',
      id: Date.now(),
      result: {
        isError: false,
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: 'healthy',
            prover: 'jolt-atlas',
            _simulated: true,
          }),
        }],
      },
    };
  }

  return { error: 'Unknown method' };
}
