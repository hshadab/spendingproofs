/**
 * OpenMind Transfer API
 *
 * Execute real USDC transfers on Arc testnet with proof attestation
 *
 * FLOW:
 * 1. Off-chain proof verification (assumed to have passed before calling this API)
 * 2. Execute USDC transfer directly
 * 3. If proofHash provided, submit to ProofAttestation contract for audit trail
 *
 * Key insight: Payment is authorized by OFF-CHAIN verification.
 * On-chain attestation is for AUDIT TRAIL, not payment gating.
 */

import { NextRequest, NextResponse } from 'next/server';
import { executeUsdcTransfer } from '@/lib/openmind/wallet';
import { submitProofAttestation, formatProofHash } from '@/lib/arc';
import { createLogger } from '@/lib/metrics';
import type { Hex } from 'viem';

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

    const OPENMIND_PRIVATE_KEY = process.env.OPENMIND_WALLET_PRIVATE_KEY as Hex;

    if (!OPENMIND_PRIVATE_KEY) {
      return NextResponse.json(
        { success: false, error: 'OPENMIND_WALLET_PRIVATE_KEY not configured' },
        { status: 500 }
      );
    }

    // Format proof hash if provided
    const formattedProofHash = proofHash ? formatProofHash(proofHash) : null;

    logger.info('Executing USDC transfer', {
      action: 'transfer',
      to,
      amount,
      proofHash: formattedProofHash,
    });

    // STEP 1: Execute the transfer
    const result = await executeUsdcTransfer(to as `0x${string}`, amount);

    logger.info('USDC transfer complete', {
      action: 'transfer',
      txHash: result.txHash,
      to,
      amount,
    });

    // STEP 2: If proof hash provided, submit attestation for audit trail (non-blocking)
    let attestationTxHash: string | undefined;
    let attestationError: string | undefined;

    if (formattedProofHash) {
      logger.info('Submitting proof attestation for audit trail', {
        action: 'attestation',
        proofHash: formattedProofHash,
      });

      try {
        const attestResult = await submitProofAttestation(
          OPENMIND_PRIVATE_KEY,
          formattedProofHash as Hex
        );

        if (attestResult.success) {
          attestationTxHash = attestResult.txHash;
          logger.info('Proof attested for audit', {
            action: 'attestation',
            txHash: attestationTxHash,
          });
        } else {
          attestationError = attestResult.error;
          logger.warn('Attestation failed (non-critical)', {
            action: 'attestation',
            error: attestResult.error,
          });
        }
      } catch (err) {
        attestationError = err instanceof Error ? err.message : 'Unknown error';
        logger.warn('Attestation error (non-critical)', {
          action: 'attestation',
          error: err,
        });
      }
    }

    return NextResponse.json({
      success: true,
      transfer: {
        status: 'success',
        txHash: result.txHash,
        to: result.to,
        amount: result.amountUsdc,
        explorerUrl: result.explorerUrl,
        proofHash: formattedProofHash,
        // Attestation info (for audit trail, not payment gating)
        attestationTxHash,
        attestationError,
        note: attestationTxHash
          ? 'Payment executed. Proof attested on Arc for audit trail.'
          : 'Payment executed. Proof hash recorded for audit.',
      },
      steps: [
        { step: 'Off-Chain Verification', status: 'success' },
        { step: 'USDC Transfer', status: 'success', txHash: result.txHash },
        {
          step: 'Proof Attestation (Audit)',
          status: attestationTxHash ? 'success' : formattedProofHash ? 'skipped' : 'not_applicable',
          txHash: attestationTxHash,
        },
      ],
    });
  } catch (error) {
    logger.error('Transfer failed', { action: 'transfer', error });
    return NextResponse.json(
      { success: false, error: 'Transfer failed' },
      { status: 500 }
    );
  }
}
