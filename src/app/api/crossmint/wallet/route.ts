import { NextRequest, NextResponse } from 'next/server';
import {
  getOrCreateAgentWallet,
  getWalletBalance,
  listWallets,
} from '@/lib/crossmint';
import { createLogger } from '@/lib/metrics';
import { createErrorResponse } from '@/lib/validation';

const logger = createLogger('api:crossmint:wallet');

/**
 * GET /api/wallet - Get agent wallet info and balance
 * GET /api/wallet?list=true - List all wallets
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const listAll = searchParams.get('list') === 'true';

    if (listAll) {
      const wallets = await listWallets();
      return NextResponse.json({ success: true, wallets });
    }

    // Get or create the demo agent wallet
    const wallet = await getOrCreateAgentWallet();

    // Get balance
    let balances: { currency: string; amount: string; chain: string }[] = [];
    try {
      balances = await getWalletBalance(wallet.address);
    } catch (error) {
      logger.warn('Could not fetch balance', { action: 'get_balance', error });
    }

    // Find USDC balance
    const usdcBalance = balances.find(b =>
      b.currency.toLowerCase().includes('usdc')
    );

    return NextResponse.json({
      success: true,
      wallet: {
        address: wallet.address,
        type: wallet.type,
        chain: wallet.chain,
      },
      balance: {
        usdc: usdcBalance?.amount || '0',
        all: balances,
      },
    });
  } catch (error) {
    logger.error('Wallet API error', { action: 'get_wallet', error });
    return NextResponse.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Unknown error',
        'INTERNAL_ERROR'
      ),
      { status: 500 }
    );
  }
}

/**
 * POST /api/wallet - Create a new wallet
 */
export async function POST() {
  try {
    const wallet = await getOrCreateAgentWallet();

    logger.info('Wallet created/retrieved', { action: 'create_wallet', address: wallet.address });

    return NextResponse.json({
      success: true,
      wallet: {
        address: wallet.address,
        type: wallet.type,
        chain: wallet.chain,
        createdAt: wallet.createdAt,
      },
    });
  } catch (error) {
    logger.error('Create wallet error', { action: 'create_wallet', error });
    return NextResponse.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Unknown error',
        'INTERNAL_ERROR'
      ),
      { status: 500 }
    );
  }
}
