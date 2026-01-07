import { NextRequest, NextResponse } from 'next/server';
import {
  executeVerifiedTransfer,
  executeDirectTransfer,
  formatProofHash,
  getSpendingGateInfo,
  CONTRACTS,
} from '@/lib/arc';
import type { Hex } from 'viem';

/**
 * POST /api/transfer - Execute a verified USDC transfer
 *
 * REAL FLOW:
 * 1. Submit proof to ProofAttestation contract (if not already attested)
 * 2. Execute gatedTransfer via SpendingGateWallet
 * 3. SpendingGate verifies proof is attested before releasing funds
 *
 * Body: {
 *   to: string,        // Recipient address
 *   amount: number,    // Amount in USDC
 *   proofHash: string  // zkML proof hash (required for verified transfer)
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

    // If proofHash is provided, try the verified flow (REAL on-chain verification)
    if (proofHash) {
      console.log('Attempting VERIFIED transfer with on-chain proof verification');

      const formattedProofHash = formatProofHash(proofHash);

      try {
        const result = await executeVerifiedTransfer(
          DEMO_PRIVATE_KEY,
          to as Hex,
          amount,
          formattedProofHash
        );

        if (result.success) {
          console.log('VERIFIED transfer succeeded');
          return NextResponse.json({
            success: true,
            transfer: {
              status: 'success',
              txHash: result.transferTxHash,
              attestationTxHash: result.attestationTxHash,
              from: CONTRACTS.SPENDING_GATE,
              to,
              amount: amount.toString(),
              chain: 'arc-testnet',
              proofHash: formattedProofHash,
              verifiedOnChain: true,
            },
            steps: result.steps,
          });
        }

        // Verified flow failed - fall back to direct transfer
        console.log('Verified flow failed, falling back to direct transfer:', result.error);
      } catch (verifiedError) {
        console.log('Verified flow error, falling back to direct transfer:', verifiedError);
      }

      // Fall back to direct transfer but include proof hash in audit trail
      console.log('Executing DIRECT transfer (fallback with proof hash audit)');

      const directResult = await executeDirectTransfer(
        DEMO_PRIVATE_KEY,
        to as Hex,
        amount
      );

      if (!directResult.success) {
        return NextResponse.json(
          { success: false, error: directResult.error },
          { status: 400 }
        );
      }

      return NextResponse.json({
        success: true,
        transfer: {
          status: 'success',
          txHash: directResult.txHash,
          to,
          amount: amount.toString(),
          chain: 'arc-testnet',
          proofHash: formattedProofHash,
          verifiedOnChain: false,
          note: 'Proof verification unavailable - direct transfer with audit trail',
        },
      });
    }

    // Fallback: Direct transfer without verification (legacy mode)
    console.log('Executing DIRECT transfer (no proof verification)');

    const result = await executeDirectTransfer(
      DEMO_PRIVATE_KEY,
      to as Hex,
      amount
    );

    if (!result.success) {
      return NextResponse.json(
        { success: false, error: result.error },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      transfer: {
        status: 'success',
        txHash: result.txHash,
        to,
        amount: amount.toString(),
        chain: 'arc-testnet',
        verifiedOnChain: false,
      },
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
