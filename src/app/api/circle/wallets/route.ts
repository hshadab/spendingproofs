import { NextRequest, NextResponse } from 'next/server';
import { parseCircleApiKey, getCircleApiUrl, CircleAPIError } from '@/lib/circle';

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

// GET /api/circle/wallets - List wallets
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const walletSetId = searchParams.get('walletSetId');

    let endpoint = '/wallets';
    if (walletSetId) {
      endpoint += `?walletSetId=${walletSetId}`;
    }

    const data = await circleRequest(endpoint);
    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof CircleAPIError) {
      return NextResponse.json(
        { error: error.message, code: error.code },
        { status: error.statusCode }
      );
    }
    return NextResponse.json(
      { error: 'Failed to fetch wallets' },
      { status: 500 }
    );
  }
}

// POST /api/circle/wallets - Create wallet
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { walletSetId, blockchain, accountType, metadata } = body;

    if (!walletSetId || !blockchain) {
      return NextResponse.json(
        { error: 'walletSetId and blockchain are required' },
        { status: 400 }
      );
    }

    const data = await circleRequest('/wallets', 'POST', {
      walletSetId,
      blockchains: [blockchain],
      count: 1,
      accountType: accountType || 'EOA',
      metadata: metadata || [],
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
      { error: 'Failed to create wallet' },
      { status: 500 }
    );
  }
}
