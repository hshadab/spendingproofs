import { NextRequest, NextResponse } from 'next/server';
import { parseCircleApiKey, getCircleApiUrl, CircleAPIError, formatUsdcAmount, getUsdcAddress } from '@/lib/circle';
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

// POST /api/circle/transfer - Execute USDC transfer
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      walletId,
      destinationAddress,
      amount,
      blockchain,
      feeLevel = 'MEDIUM',
    } = body;

    if (!walletId || !destinationAddress || !amount || !blockchain) {
      return NextResponse.json(
        { error: 'walletId, destinationAddress, amount, and blockchain are required' },
        { status: 400 }
      );
    }

    const usdcAddress = getUsdcAddress(blockchain);
    if (!usdcAddress) {
      return NextResponse.json(
        { error: `USDC not supported on blockchain: ${blockchain}` },
        { status: 400 }
      );
    }

    // Create transfer transaction
    const data = await circleRequest('/transactions/transfer', 'POST', {
      idempotencyKey: randomUUID(),
      walletId,
      tokenId: usdcAddress,
      destinationAddress,
      amounts: [formatUsdcAmount(amount)],
      feeLevel,
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
      { error: 'Failed to execute transfer' },
      { status: 500 }
    );
  }
}

// GET /api/circle/transfer?id=xxx - Get transfer status
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const transactionId = searchParams.get('id');

    if (!transactionId) {
      return NextResponse.json(
        { error: 'Transaction ID is required' },
        { status: 400 }
      );
    }

    const data = await circleRequest(`/transactions/${transactionId}`);
    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof CircleAPIError) {
      return NextResponse.json(
        { error: error.message, code: error.code },
        { status: error.statusCode }
      );
    }
    return NextResponse.json(
      { error: 'Failed to get transfer status' },
      { status: 500 }
    );
  }
}
