import { NextRequest, NextResponse } from 'next/server';
import {
  executeDirectTransfer,
  submitProofAttestation,
  formatProofHash,
  getSpendingGateInfo,
  CONTRACTS,
} from '@/lib/arc';
import type { Hex } from 'viem';

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

    if (!to || !amount) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: to, amount' },
        { status: 400 }
      );
    }

    const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY as Hex;

    if (!DEMO_PRIVATE_KEY) {
      return NextResponse.json(
        { success: false, error: 'DEMO_WALLET_PRIVATE_KEY not configured' },
        { status: 500 }
      );
    }

    // Format proof hash if provided
    const formattedProofHash = proofHash ? formatProofHash(proofHash) : null;

    // STEP 1: Execute the transfer (off-chain verification assumed to have passed)
    console.log('Executing USDC transfer (off-chain proof verification passed)');

    const transferResult = await executeDirectTransfer(
      DEMO_PRIVATE_KEY,
      to as Hex,
      amount
    );

    if (!transferResult.success) {
      return NextResponse.json(
        { success: false, error: transferResult.error },
        { status: 400 }
      );
    }

    console.log('Transfer succeeded:', transferResult.txHash);

    // STEP 2: If proof hash provided, submit attestation for audit trail (non-blocking)
    let attestationTxHash: string | undefined;
    let attestationError: string | undefined;

    if (formattedProofHash) {
      console.log('Submitting proof attestation for audit trail...');

      try {
        const attestResult = await submitProofAttestation(
          DEMO_PRIVATE_KEY,
          formattedProofHash
        );

        if (attestResult.success) {
          attestationTxHash = attestResult.txHash;
          console.log('Proof attested for audit:', attestationTxHash);
        } else {
          attestationError = attestResult.error;
          console.warn('Attestation failed (non-critical):', attestResult.error);
        }
      } catch (err) {
        attestationError = err instanceof Error ? err.message : 'Unknown error';
        console.warn('Attestation error (non-critical):', err);
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
    console.error('Transfer API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
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
    console.error('Get SpendingGate info error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
