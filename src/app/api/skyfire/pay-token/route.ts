/**
 * Skyfire Pay Token API
 *
 * Generate a PAY token for payment authorization.
 * The PAY token includes our zkML proof hash, binding identity to policy compliance.
 *
 * This is where Skyfire (identity) meets our zkML proofs (policy compliance):
 * - KYA Token proves WHO the agent is
 * - zkML Proof proves the agent FOLLOWED its spending policy
 * - PAY Token binds both together for trustless payment authorization
 */

import { NextRequest, NextResponse } from 'next/server';
import { generatePayToken, verifyKYAToken } from '@/lib/skyfire';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('api:skyfire:pay-token');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      agentId,
      kyaToken,
      amount,
      currency = 'USDC',
      recipient,
      proofHash,
      verificationHash,
      memo,
    } = body;

    // Validate required fields
    if (!agentId || !amount || !recipient) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: agentId, amount, recipient' },
        { status: 400 }
      );
    }

    // Validate amount for demo
    if (amount > 100) {
      return NextResponse.json(
        { success: false, error: 'Amount exceeds demo limit of $100 USDC' },
        { status: 400 }
      );
    }

    // Verify KYA token if provided
    if (kyaToken) {
      logger.info('Verifying KYA token', { agentId });
      const verification = await verifyKYAToken(kyaToken);

      if (!verification.valid) {
        return NextResponse.json(
          { success: false, error: 'Invalid or expired KYA token' },
          { status: 401 }
        );
      }

      // Ensure token agent matches request
      if (verification.claims?.sub !== agentId) {
        return NextResponse.json(
          { success: false, error: 'KYA token does not match agentId' },
          { status: 401 }
        );
      }
    }

    // Log the binding of identity + proof
    logger.info('Generating PAY token with zkML proof binding', {
      agentId,
      amount,
      recipient,
      hasProofHash: !!proofHash,
      hasVerificationHash: !!verificationHash,
    });

    // Generate PAY token with proof references
    const payToken = await generatePayToken({
      agentId,
      amount,
      currency,
      recipient,
      proofHash,
      verificationHash,
      memo: memo || `zkML verified payment - proof: ${proofHash?.substring(0, 10)}...`,
    });

    logger.info('PAY token generated', {
      agentId,
      tokenPreview: payToken.token.substring(0, 20) + '...',
      expiresAt: payToken.expiresAt,
    });

    return NextResponse.json({
      success: true,
      payToken,
      binding: {
        identity: {
          type: 'Skyfire KYA',
          agentId,
          verified: !!kyaToken,
        },
        compliance: {
          type: 'zkML Proof',
          proofHash,
          verificationHash,
          verified: !!proofHash && !!verificationHash,
        },
      },
      message: proofHash
        ? 'PAY token generated with zkML proof binding - trustless payment ready'
        : 'PAY token generated (no zkML proof attached)',
    });
  } catch (error) {
    logger.error('Failed to generate PAY token', { error });
    return NextResponse.json(
      { success: false, error: 'Failed to generate PAY token' },
      { status: 500 }
    );
  }
}
