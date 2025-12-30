import { NextRequest, NextResponse } from 'next/server';
import { parseCircleApiKey, getCircleApiUrl, CircleAPIError } from '@/lib/circle';
import { randomUUID } from 'crypto';

// Get Circle API key from environment
function getCircleCredentials() {
  const apiKey = process.env.CIRCLE_API_KEY;
  if (!apiKey) {
    throw new CircleAPIError('Circle API key not configured', 500, 'CONFIG_ERROR');
  }
  return parseCircleApiKey(apiKey);
}

// Make authenticated request to Circle API
async function circleRequest(
  endpoint: string,
  method: 'GET' | 'POST' = 'GET',
  body?: unknown
) {
  const { apiKey, isTest } = getCircleCredentials();
  const baseUrl = getCircleApiUrl(isTest);

  const response = await fetch(`${baseUrl}${endpoint}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new CircleAPIError(
      data.message || 'Circle API error',
      response.status,
      data.code
    );
  }

  return data;
}

// GET /api/circle/wallet-sets - List wallet sets
export async function GET() {
  try {
    const data = await circleRequest('/walletSets');
    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof CircleAPIError) {
      return NextResponse.json(
        { error: error.message, code: error.code },
        { status: error.statusCode }
      );
    }
    return NextResponse.json(
      { error: 'Failed to fetch wallet sets' },
      { status: 500 }
    );
  }
}

// POST /api/circle/wallet-sets - Create wallet set
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name } = body;

    if (!name) {
      return NextResponse.json(
        { error: 'name is required' },
        { status: 400 }
      );
    }

    const data = await circleRequest('/walletSets', 'POST', {
      idempotencyKey: randomUUID(),
      name,
    });

    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof CircleAPIError) {
      return NextResponse.json(
        { error: error.message, code: error.code },
        { status: error.statusCode }
      );
    }
    return NextResponse.json(
      { error: 'Failed to create wallet set' },
      { status: 500 }
    );
  }
}
