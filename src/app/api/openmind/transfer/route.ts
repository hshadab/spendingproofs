/**
 * OpenMind Transfer API
 *
 * Execute real USDC transfers on Arc testnet
 */

import { NextRequest, NextResponse } from 'next/server';
import { executeUsdcTransfer } from '@/lib/openmind/wallet';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('api:openmind:transfer');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { to, amount, proofHash } = body;

    if (!to || !amount) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: to, amount' },
        { status: 400 }
      );
    }

    // Validate amount is reasonable for demo
    if (amount > 5) {
      return NextResponse.json(
        { success: false, error: 'Amount exceeds demo limit of $5 USDC' },
        { status: 400 }
      );
    }

    logger.info('Executing USDC transfer', {
      action: 'transfer',
      to,
      amount,
      proofHash,
    });

    const result = await executeUsdcTransfer(to as `0x${string}`, amount);

    logger.info('USDC transfer complete', {
      action: 'transfer',
      txHash: result.txHash,
      to,
      amount,
    });

    return NextResponse.json({
      success: true,
      transfer: {
        txHash: result.txHash,
        to: result.to,
        amount: result.amountUsdc,
        explorerUrl: result.explorerUrl,
      },
    });
  } catch (error) {
    logger.error('Transfer failed', { action: 'transfer', error });
    return NextResponse.json(
      { success: false, error: 'Transfer failed' },
      { status: 500 }
    );
  }
}
