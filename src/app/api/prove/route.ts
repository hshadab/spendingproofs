import { NextRequest, NextResponse } from 'next/server';
import { verifyProveRequest, isSignatureAuthEnabled } from '@/lib/signatureAuth';
import { createLogger } from '@/lib/metrics';
import { API_CONFIG } from '@/lib/config';
import { createErrorResponse } from '@/lib/validation';
import type { SignedProveRequest } from '@/lib/types';

const logger = createLogger('api:prove');
const PROVER_BACKEND_URL = API_CONFIG.proverBackendUrl;

export async function POST(request: NextRequest) {
  try {
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

    // Forward request to prover backend (only send inputs and tag)
    const response = await fetch(`${PROVER_BACKEND_URL}/prove`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs: body.inputs, tag: body.tag }),
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
