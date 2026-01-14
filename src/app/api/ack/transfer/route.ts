/**
 * ACK Transfer API
 *
 * Execute real USDC transfers on Arc testnet via SpendingGateWallet
 *
 * FLOW (Option B - Full SpendingGateWallet):
 * 1. Compute verificationHash = keccak256(proofHash, decision, timestamp)
 * 2. Submit verificationHash to ProofAttestation contract (required first)
 * 3. Execute gatedTransfer via SpendingGateWallet (checks attestation)
 *
 * Key insight: We attest the VERIFICATION RESULT, not just the proof.
 * verificationHash captures: proof existed + was verified + decision was approved.
 * This is what should gate payment - proof that verification passed.
 */

import { NextRequest, NextResponse } from 'next/server';
import { keccak256, encodeAbiParameters, parseAbiParameters } from 'viem';
import {
  submitProofAttestation,
  executeGatedTransfer,
  isProofAttested,
  formatProofHash,
  getExplorerUrl,
} from '@/lib/arc';
import { createLogger } from '@/lib/metrics';
import type { Hex } from 'viem';

const logger = createLogger('api:ack:transfer');

/**
 * Compute verificationHash from proof verification result
 * This hash represents "proof was verified and decision was X"
 */
function computeVerificationHash(
  proofHash: Hex,
  decision: boolean,
  confidence: number,
  timestamp: number
): Hex {
  // Encode: proofHash + decision (bool) + confidence (scaled to uint256) + timestamp
  const encoded = encodeAbiParameters(
    parseAbiParameters('bytes32, bool, uint256, uint256'),
    [
      proofHash,
      decision,
      BigInt(Math.round(confidence * 10000)), // Scale confidence to basis points
      BigInt(timestamp),
    ]
  );
  return keccak256(encoded);
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      to,
      amount,
      proofHash,
      agentDid,
      // Verification result - this is what we attest
      decision,        // boolean: shouldBuy
      confidence,      // number: 0-1
      verifiedAt,      // timestamp when verification passed
    } = body;

    if (!to || !amount || !proofHash) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: to, amount, proofHash' },
        { status: 400 }
      );
    }

    // Require verification data for attestation
    if (decision === undefined || confidence === undefined) {
      return NextResponse.json(
        { success: false, error: 'Missing verification data: decision, confidence required' },
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

    const ACK_PRIVATE_KEY = process.env.ACK_WALLET_PRIVATE_KEY as Hex;

    if (!ACK_PRIVATE_KEY) {
      return NextResponse.json(
        { success: false, error: 'ACK_WALLET_PRIVATE_KEY not configured' },
        { status: 500 }
      );
    }

    // Format proof hash
    const formattedProofHash = formatProofHash(proofHash);

    // Compute verificationHash - this is what we attest
    // It captures: proof existed + was verified + decision result
    const timestamp = verifiedAt || Math.floor(Date.now() / 1000);
    const verificationHash = computeVerificationHash(
      formattedProofHash,
      decision,
      confidence,
      timestamp
    );

    logger.info('Starting ACK gated transfer flow', {
      action: 'ack_transfer',
      to,
      amount,
      proofHash: formattedProofHash,
      verificationHash,
      decision,
      confidence,
      agentDid,
    });

    const steps: { step: string; status: 'success' | 'failed' | 'skipped'; txHash?: string; hash?: string }[] = [];

    // STEP 1: Check if verificationHash is already attested
    let attestationTxHash: string | undefined;
    const alreadyAttested = await isProofAttested(verificationHash);

    if (alreadyAttested) {
      logger.info('Verification already attested, skipping attestation', {
        action: 'attestation',
        verificationHash,
      });
      steps.push({ step: 'Verification Attestation', status: 'skipped', hash: verificationHash });
    } else {
      // Submit verificationHash attestation (REQUIRED for gated transfer)
      logger.info('Submitting verification attestation', {
        action: 'attestation',
        verificationHash,
        proofHash: formattedProofHash,
        decision,
      });

      // Include metadata about what was verified
      const metadata = encodeAbiParameters(
        parseAbiParameters('bytes32, bool, uint256'),
        [formattedProofHash, decision, BigInt(Math.round(confidence * 10000))]
      );

      const attestResult = await submitProofAttestation(
        ACK_PRIVATE_KEY,
        verificationHash,
        metadata
      );

      if (!attestResult.success) {
        logger.error('Attestation failed', {
          action: 'attestation',
          error: attestResult.error,
        });
        steps.push({ step: 'Verification Attestation', status: 'failed' });
        return NextResponse.json({
          success: false,
          error: `Attestation failed: ${attestResult.error}`,
          steps,
        });
      }

      attestationTxHash = attestResult.txHash;
      steps.push({
        step: 'Verification Attestation',
        status: 'success',
        txHash: attestationTxHash,
        hash: verificationHash,
      });

      logger.info('Verification attested successfully', {
        action: 'attestation',
        txHash: attestationTxHash,
        verificationHash,
      });

      // Wait a moment for attestation to be indexed
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // STEP 2: Execute gated transfer via SpendingGateWallet
    // Uses verificationHash (not proofHash) to check attestation
    logger.info('Executing gated transfer', {
      action: 'gated_transfer',
      to,
      amount,
      verificationHash,
    });

    const transferResult = await executeGatedTransfer(
      ACK_PRIVATE_KEY,
      to as Hex,
      amount,
      verificationHash, // Use verificationHash, not proofHash
      3600 // 1 hour expiry
    );

    if (!transferResult.success) {
      logger.error('Gated transfer failed', {
        action: 'gated_transfer',
        error: transferResult.error,
      });
      steps.push({ step: 'Gated Transfer', status: 'failed' });
      return NextResponse.json({
        success: false,
        error: `Gated transfer failed: ${transferResult.error}`,
        attestationTxHash,
        verificationHash,
        steps,
      });
    }

    steps.push({ step: 'Gated Transfer', status: 'success', txHash: transferResult.txHash });

    logger.info('Gated transfer successful', {
      action: 'gated_transfer',
      txHash: transferResult.txHash,
    });

    return NextResponse.json({
      success: true,
      transfer: {
        status: 'success',
        txHash: transferResult.txHash,
        to,
        amount,
        explorerUrl: getExplorerUrl(transferResult.txHash!),
        // Include both hashes for transparency
        proofHash: formattedProofHash,
        verificationHash,
        // Attestation details
        attestationTxHash,
        attestationExplorerUrl: attestationTxHash ? getExplorerUrl(attestationTxHash) : undefined,
        // Verification data that was attested
        verification: {
          decision,
          confidence,
          timestamp,
        },
        note: 'Verification result attested on-chain (not just proof). Gated transfer checked attestation before releasing funds.',
      },
      steps,
    });
  } catch (error) {
    logger.error('ACK transfer failed', { action: 'ack_transfer', error });
    return NextResponse.json(
      { success: false, error: 'Transfer failed' },
      { status: 500 }
    );
  }
}
