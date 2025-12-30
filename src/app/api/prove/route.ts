import { NextRequest, NextResponse } from 'next/server';

const PROVER_BACKEND_URL = process.env.PROVER_BACKEND_URL || 'http://localhost:3001';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Forward request to prover backend
    const response = await fetch(`${PROVER_BACKEND_URL}/prove`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
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
    console.error('Prover proxy error:', error);

    // Check if it's a connection error (prover unavailable)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        {
          success: false,
          error: 'Prover service unavailable. Please ensure the prover is running.',
          code: 'PROVER_UNAVAILABLE'
        },
        { status: 503 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
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
