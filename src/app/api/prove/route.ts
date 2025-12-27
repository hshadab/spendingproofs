/**
 * POST /api/prove
 *
 * Proxy to JOLT-Atlas prover service for generating zkML proofs.
 */

import { NextRequest, NextResponse } from 'next/server';

const JOLT_ATLAS_URL = process.env.NEXT_PUBLIC_JOLT_ATLAS_URL || 'http://localhost:3001';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { inputs, tag = 'spending' } = body;

    if (!inputs || !Array.isArray(inputs)) {
      return NextResponse.json(
        { success: false, error: 'Missing or invalid inputs array' },
        { status: 400 }
      );
    }

    // Forward to JOLT-Atlas prover
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

    const response = await fetch(`${JOLT_ATLAS_URL}/prove`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: 'spending-model',
        inputs,
        tag,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        {
          success: false,
          error: `Prover service error: ${response.status} - ${error}`,
          generationTimeMs: 0,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          success: false,
          error: 'Proof generation timed out after 120 seconds',
          generationTimeMs: 120000,
        },
        { status: 504 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        generationTimeMs: 0,
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Health check - forward to prover
  try {
    const response = await fetch(`${JOLT_ATLAS_URL}/health`);
    if (!response.ok) {
      return NextResponse.json(
        { status: 'error', message: 'Prover service unavailable' },
        { status: 503 }
      );
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: 'error', message: 'Prover service unavailable' },
      { status: 503 }
    );
  }
}
