/**
 * OpenMind Wallet API
 *
 * Get real wallet balance from Arc testnet
 */

import { NextResponse } from 'next/server';
import { getWalletInfo, getWalletBalance, simulateUsdcTransfer } from '@/lib/openmind/wallet';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('api:openmind:wallet');

export async function GET() {
  try {
    const walletInfo = await getWalletInfo();

    logger.info('Wallet info fetched', {
      action: 'get_wallet',
      address: walletInfo.address,
      balance: walletInfo.balanceUsdc,
    });

    return NextResponse.json({
      success: true,
      wallet: walletInfo,
    });
  } catch (error) {
    logger.error('Failed to fetch wallet info', { action: 'get_wallet', error });
    return NextResponse.json(
      { success: false, error: 'Failed to fetch wallet info' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, to, amount } = body;

    if (action === 'simulate') {
      const result = await simulateUsdcTransfer(to as `0x${string}`, amount);
      return NextResponse.json({ success: true, simulation: result });
    }

    if (action === 'balance') {
      const balance = await getWalletBalance();
      return NextResponse.json({ success: true, balance });
    }

    return NextResponse.json(
      { success: false, error: 'Unknown action' },
      { status: 400 }
    );
  } catch (error) {
    logger.error('Wallet action failed', { action: 'wallet_action', error });
    return NextResponse.json(
      { success: false, error: 'Wallet action failed' },
      { status: 500 }
    );
  }
}
