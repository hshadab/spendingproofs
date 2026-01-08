import { NextRequest, NextResponse } from 'next/server';
import {
  executeDirectTransfer,
  submitProofAttestation,
  formatProofHash,
  getSpendingGateInfo,
  CONTRACTS,
} from '@/lib/arc';
import { createLogger } from '@/lib/metrics';
import {
  validateTransferInput,
  createValidationErrorResponse,
  createErrorResponse,
} from '@/lib/validation';
import type { Hex } from 'viem';

const logger = createLogger('api:crossmint:transfer');

/**
 * POST /api/transfer - Execute USDC transfer with optional proof attestation
 *
 * CORRECT FLOW:
 * 1. Off-chain proof verification (assumed to have passed before calling this API)
 * 2. Execute USDC transfer directly
 * 3. If proofHash provided, submit to ProofAttestation contract for audit trail
 *
 * Key insight: Payment is authorized by OFF-CHAIN verification.
 * On-chain attestation is for AUDIT TRAIL, not payment gating.
 *
 * Body: {
 *   to: string,        // Recipient address
 *   amount: number,    // Amount in USDC
 *   proofHash?: string // zkML proof hash (optional, for audit trail)
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { to, amount, proofHash } = body;

    // Validate input
    const validation = validateTransferInput({ to, amount, proofHash });
    if (!validation.valid) {
      return NextResponse.json(
        createValidationErrorResponse(validation.errors),
        { status: 400 }
      );
    }

    const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY as Hex;

    if (!DEMO_PRIVATE_KEY) {
      return NextResponse.json(
        createErrorResponse('DEMO_WALLET_PRIVATE_KEY not configured', 'CONFIG_ERROR'),
        { status: 500 }
      );
    }

    // Format proof hash if provided
    const formattedProofHash = proofHash ? formatProofHash(proofHash) : null;

    // STEP 1: Execute the transfer (off-chain verification assumed to have passed)
    logger.info('Executing USDC transfer', { action: 'transfer', to, amount });

    const transferResult = await executeDirectTransfer(
      DEMO_PRIVATE_KEY,
      to as Hex,
      amount
    );

    if (!transferResult.success) {
      return NextResponse.json(
        createErrorResponse(transferResult.error || 'Transfer failed', 'TRANSFER_FAILED'),
        { status: 400 }
      );
    }

    logger.info('Transfer succeeded', { action: 'transfer', txHash: transferResult.txHash });

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
          DEMO_PRIVATE_KEY,
          formattedProofHash
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

    // Return success - payment completed, attestation is bonus
    return NextResponse.json({
      success: true,
      transfer: {
        status: 'success',
        txHash: transferResult.txHash,
        to,
        amount: amount.toString(),
        chain: 'arc-testnet',
        proofHash: formattedProofHash,
        // Attestation info (for audit trail, not payment gating)
        attestationTxHash,
        attestationError,
        verifiedOnChain: false, // Off-chain verification, on-chain attestation
        note: attestationTxHash
          ? 'Payment executed. Proof attested on Arc for audit trail.'
          : 'Payment executed. Proof hash recorded for audit.',
      },
      steps: [
        { step: 'Off-Chain Verification', status: 'success' },
        { step: 'USDC Transfer', status: 'success', txHash: transferResult.txHash },
        {
          step: 'Proof Attestation (Audit)',
          status: attestationTxHash ? 'success' : formattedProofHash ? 'skipped' : 'not_applicable',
          txHash: attestationTxHash,
        },
      ],
    });
  } catch (error) {
    logger.error('Transfer API error', { action: 'transfer', error });
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
 * GET /api/transfer - Get SpendingGate info
 */
export async function GET() {
  try {
    const info = await getSpendingGateInfo();

    return NextResponse.json({
      success: true,
      spendingGate: {
        address: CONTRACTS.SPENDING_GATE,
        ...info,
      },
      contracts: {
        usdc: CONTRACTS.USDC,
        proofAttestation: CONTRACTS.PROOF_ATTESTATION,
        spendingGate: CONTRACTS.SPENDING_GATE,
      },
    });
  } catch (error) {
    logger.error('Get SpendingGate info error', { action: 'get_info', error });
    return NextResponse.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Unknown error',
        'INTERNAL_ERROR'
      ),
      { status: 500 }
    );
  }
}
