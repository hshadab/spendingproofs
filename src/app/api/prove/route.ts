import { NextRequest, NextResponse } from 'next/server';
import { verifyProveRequest, isSignatureAuthEnabled } from '@/lib/signatureAuth';
import { createLogger } from '@/lib/metrics';
import { API_CONFIG, AUTH_CONFIG, RATE_LIMIT_CONFIG } from '@/lib/config';
import { createErrorResponse } from '@/lib/validation';
import type { SignedProveRequest } from '@/lib/types';

const logger = createLogger('api:prove');
const PROVER_BACKEND_URL = API_CONFIG.proverBackendUrl;

/**
 * In-memory sliding window rate limiter.
 * Tracks request timestamps per IP and prunes expired entries.
 */
const rateLimitMap = new Map<string, number[]>();

function isRateLimited(ip: string): { limited: boolean; retryAfterMs: number } {
  const now = Date.now();
  const { maxRequests, windowMs } = RATE_LIMIT_CONFIG;

  let timestamps = rateLimitMap.get(ip) || [];
  // Prune timestamps outside the window
  timestamps = timestamps.filter((t) => now - t < windowMs);

  if (timestamps.length >= maxRequests) {
    const oldestInWindow = timestamps[0];
    const retryAfterMs = windowMs - (now - oldestInWindow);
    rateLimitMap.set(ip, timestamps);
    return { limited: true, retryAfterMs };
  }

  timestamps.push(now);
  rateLimitMap.set(ip, timestamps);
  return { limited: false, retryAfterMs: 0 };
}

function getClientIp(request: NextRequest): string {
  return (
    request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ||
    request.headers.get('x-real-ip') ||
    'unknown'
  );
}

/**
 * Verify API key for AgentCore Gateway requests.
 * Only enforced when GATEWAY_API_KEY is configured.
 */
function verifyGatewayApiKey(request: NextRequest): { valid: boolean; error?: string } {
  const configuredKey = AUTH_CONFIG.gatewayApiKey;

  // If no API key configured, allow all requests
  if (!configuredKey) {
    return { valid: true };
  }

  const providedKey = request.headers.get('x-api-key');

  if (!providedKey) {
    return { valid: false, error: 'API key required' };
  }

  if (providedKey !== configuredKey) {
    return { valid: false, error: 'Invalid API key' };
  }

  return { valid: true };
}

export async function POST(request: NextRequest) {
  try {
    // Rate limit check
    const clientIp = getClientIp(request);
    const rateCheck = isRateLimited(clientIp);
    if (rateCheck.limited) {
      const retryAfterSec = Math.ceil(rateCheck.retryAfterMs / 1000);
      return NextResponse.json(
        { success: false, error: 'Too many requests. Please try again later.' },
        { status: 429, headers: { 'Retry-After': String(retryAfterSec) } }
      );
    }

    // Verify API key for AgentCore Gateway requests (if configured)
    const apiKeyResult = verifyGatewayApiKey(request);
    if (!apiKeyResult.valid) {
      return NextResponse.json(
        { success: false, error: apiKeyResult.error },
        { status: 401 }
      );
    }

    const body = await request.json();

    // Verify signature if authentication is enabled
    if (isSignatureAuthEnabled()) {
      const { inputs, tag, address, timestamp, signature } = body as SignedProveRequest;

      // Check required fields for signed request
      if (!address || !timestamp || !signature) {
        return NextResponse.json(
          {
            success: false,
            error: 'Signature authentication required. Missing: address, timestamp, or signature',
            code: 'INVALID_SIGNATURE',
          },
          { status: 401 }
        );
      }

      const verification = await verifyProveRequest({ inputs, tag, address, timestamp, signature });

      if (!verification.valid) {
        return NextResponse.json(
          {
            success: false,
            error: verification.error,
            code: verification.code,
          },
          { status: 401 }
        );
      }
    }

    // Forward request to prover backend with model_id
    const response = await fetch(`${PROVER_BACKEND_URL}/prove`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: body.model_id || 'spending-model',  // Default to spending-model
        inputs: body.inputs,
        tag: body.tag
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { success: false, error: `Prover error: ${errorText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    logger.error('Prover proxy error', {
      error,
      action: 'proxy_request',
    });

    // Check if it's a connection error (prover unavailable)
    if (error instanceof TypeError) {
      return NextResponse.json(
        createErrorResponse(
          'Prover service unavailable. Please ensure the prover is running.',
          'PROVER_UNAVAILABLE'
        ),
        { status: 503 }
      );
    }

    return NextResponse.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Unknown error',
        'INTERNAL_ERROR'
      ),
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    // Health check proxy
    const response = await fetch(`${PROVER_BACKEND_URL}/health`);

    if (!response.ok) {
      return NextResponse.json(
        { status: 'unhealthy', error: 'Prover not responding' },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: 'unavailable', error: 'Cannot connect to prover' },
      { status: 503 }
    );
  }
}
