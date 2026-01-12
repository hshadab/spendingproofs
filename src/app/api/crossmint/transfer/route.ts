import { NextRequest, NextResponse } from 'next/server';
import {
  executeDirectTransfer,
  submitProofAttestation as submitArcAttestation,
  formatProofHash as formatArcProofHash,
  getSpendingGateInfo,
  CONTRACTS,
} from '@/lib/arc';
import {
  submitProofAttestation as submitBaseSepoliaAttestation,
  formatProofHash as formatBaseSepoliaProofHash,
  BASE_SEPOLIA_CONTRACTS,
} from '@/lib/baseSepolia';
import {
  smartTransfer,
  getOrCreateAgentWallet,
  INTEGRATION_MODE,
} from '@/lib/crossmint';
import { createLogger } from '@/lib/metrics';
import {
  validateTransferInput,
  createValidationErrorResponse,
  createErrorResponse,
} from '@/lib/validation';
import type { Hex } from 'viem';

const logger = createLogger('api:crossmint:transfer');

// Prover URL for verification
const PROVER_URL = process.env.PROVER_BACKEND_URL || 'http://localhost:3002';

/**
 * Verify a zkML proof before executing transfer
 * This is the REAL verification step - not just checking a hash
 */
async function verifyProofBeforeTransfer(
  proof: string,
  modelId: string,
  modelHash: string,
  programIo: string
): Promise<{ valid: boolean; error?: string; verificationTimeMs?: number }> {
  try {
    const response = await fetch(`${PROVER_URL}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        proof,
        model_id: modelId,
        model_hash: modelHash,
        program_io: programIo,
      }),
    });

    if (!response.ok) {
      return { valid: false, error: `Prover returned ${response.status}` };
    }

    const result = await response.json();
    return {
      valid: result.valid === true,
      error: result.error,
      verificationTimeMs: result.verification_time_ms,
    };
  } catch (error) {
    logger.error('Proof verification failed', { action: 'verify_proof', error });
    return {
      valid: false,
      error: error instanceof Error ? error.message : 'Verification failed',
    };
  }
}

/**
 * POST /api/crossmint/transfer - Execute USDC transfer with zkML proof verification
 *
 * REAL INTEGRATION FLOW:
 * 1. Verify zkML proof cryptographically (via prover service)
 * 2. If verified, execute transfer via Crossmint (supported chains) or direct (Arc)
 * 3. Submit proof attestation for audit trail
 *
 * Body: {
 *   to: string,           // Recipient address
 *   amount: number,       // Amount in USDC
 *   chain?: string,       // Target chain (default: base-sepolia)
 *   proof?: string,       // Full SNARK proof (hex)
 *   proofHash?: string,   // Proof hash (for attestation)
 *   programIo?: string,   // Serialized program I/O (for verification)
 *   modelHash?: string,   // Model hash (for verification)
 *   skipVerification?: boolean // Skip verification (for demo without prover)
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      to,
      amount,
      chain = process.env.NEXT_PUBLIC_CROSSMINT_CHAIN || 'base-sepolia',
      proof,
      proofHash,
      programIo,
      modelHash,
      skipVerification = false,
    } = body;

    // Validate input
    const validation = validateTransferInput({ to, amount, proofHash });
    if (!validation.valid) {
      return NextResponse.json(
        createValidationErrorResponse(validation.errors),
        { status: 400 }
      );
    }

    const steps: { step: string; status: string; details?: string; txHash?: string; timeMs?: number }[] = [];

    // STEP 1: Verify zkML proof (if provided and not skipped)
    let proofVerified = false;
    if (proof && programIo && !skipVerification) {
      logger.info('Verifying zkML proof before transfer', { action: 'verify', proofHash });

      const verification = await verifyProofBeforeTransfer(
        proof,
        'spending-model',
        modelHash || '',
        programIo
      );

      if (verification.valid) {
        proofVerified = true;
        steps.push({
          step: 'zkML Proof Verification',
          status: 'success',
          details: 'SNARK proof cryptographically verified',
          timeMs: verification.verificationTimeMs,
        });
        logger.info('Proof verified successfully', {
          action: 'verify',
          timeMs: verification.verificationTimeMs,
        });
      } else {
        steps.push({
          step: 'zkML Proof Verification',
          status: 'failed',
          details: verification.error,
        });
        return NextResponse.json(
          createErrorResponse(
            `Proof verification failed: ${verification.error}`,
            'PROOF_INVALID'
          ),
          { status: 403 }
        );
      }
    } else if (skipVerification) {
      steps.push({
        step: 'zkML Proof Verification',
        status: 'skipped',
        details: 'Verification skipped (demo mode)',
      });
    } else {
      steps.push({
        step: 'zkML Proof Verification',
        status: 'skipped',
        details: 'No proof provided',
      });
    }

    // STEP 2: Execute transfer
    logger.info('Executing transfer', {
      action: 'transfer',
      to,
      amount,
      chain,
      mode: INTEGRATION_MODE,
    });

    let transferResult: {
      success: boolean;
      txHash?: string;
      error?: string;
      method?: string;
      crossmintTransactionId?: string;
    };

    // Use Crossmint for supported chains, direct for Arc
    const crossmintSupportedChains = ['base-sepolia', 'polygon-amoy', 'base', 'polygon', 'arbitrum', 'optimism'];

    if (crossmintSupportedChains.includes(chain) && INTEGRATION_MODE === 'crossmint') {
      // REAL Crossmint integration
      try {
        const wallet = await getOrCreateAgentWallet();
        const result = await smartTransfer(
          `evm:${chain}:${wallet.address}`,
          to,
          amount,
          chain,
          proofHash
        );
        transferResult = {
          success: result.status === 'success',
          txHash: result.txHash,
          method: 'crossmint',
          crossmintTransactionId: result.id,
        };
        steps.push({
          step: 'Crossmint Transfer',
          status: 'success',
          details: `Executed via Crossmint API on ${chain}`,
          txHash: result.txHash,
        });
      } catch (error) {
        transferResult = {
          success: false,
          error: error instanceof Error ? error.message : 'Crossmint transfer failed',
          method: 'crossmint',
        };
        steps.push({
          step: 'Crossmint Transfer',
          status: 'failed',
          details: transferResult.error,
        });
      }
    } else {
      // Direct on-chain transfer (Arc testnet)
      const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY as Hex;

      if (!DEMO_PRIVATE_KEY) {
        return NextResponse.json(
          createErrorResponse('DEMO_WALLET_PRIVATE_KEY not configured', 'CONFIG_ERROR'),
          { status: 500 }
        );
      }

      const result = await executeDirectTransfer(DEMO_PRIVATE_KEY, to as Hex, amount);
      transferResult = {
        success: result.success,
        txHash: result.txHash,
        error: result.error,
        method: 'direct',
      };

      steps.push({
        step: 'Direct Transfer',
        status: result.success ? 'success' : 'failed',
        details: result.success ? `Executed on ${chain}` : result.error,
        txHash: result.txHash,
      });
    }

    if (!transferResult.success) {
      return NextResponse.json(
        createErrorResponse(transferResult.error || 'Transfer failed', 'TRANSFER_FAILED'),
        { status: 400 }
      );
    }

    logger.info('Transfer succeeded', {
      action: 'transfer',
      txHash: transferResult.txHash,
      method: transferResult.method,
    });

    // STEP 3: Submit proof attestation for audit trail
    let attestationTxHash: string | undefined;
    let attestationError: string | undefined;

    const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY as Hex;

    if (proofHash && DEMO_PRIVATE_KEY) {
      try {
        if (chain === 'base-sepolia') {
          // Submit attestation to Base Sepolia
          const formattedHash = formatBaseSepoliaProofHash(proofHash);
          const attestResult = await submitBaseSepoliaAttestation(
            DEMO_PRIVATE_KEY,
            formattedHash
          );

          if (attestResult.success) {
            attestationTxHash = attestResult.txHash;
            steps.push({
              step: 'Proof Attestation',
              status: 'success',
              details: 'Proof hash recorded on Base Sepolia for audit',
              txHash: attestationTxHash,
            });
          } else {
            attestationError = attestResult.error;
            steps.push({
              step: 'Proof Attestation',
              status: 'skipped',
              details: attestResult.error,
            });
          }
        } else if (chain === 'arc-testnet') {
          // Submit attestation to Arc (legacy)
          const formattedHash = formatArcProofHash(proofHash);
          const attestResult = await submitArcAttestation(
            DEMO_PRIVATE_KEY,
            formattedHash
          );

          if (attestResult.success) {
            attestationTxHash = attestResult.txHash;
            steps.push({
              step: 'Proof Attestation',
              status: 'success',
              details: 'Proof hash recorded on Arc for audit',
              txHash: attestationTxHash,
            });
          } else {
            attestationError = attestResult.error;
            steps.push({
              step: 'Proof Attestation',
              status: 'skipped',
              details: attestResult.error,
            });
          }
        }
      } catch (err) {
        attestationError = err instanceof Error ? err.message : 'Unknown error';
        steps.push({
          step: 'Proof Attestation',
          status: 'skipped',
          details: attestationError,
        });
      }
    }

    // Return success
    return NextResponse.json({
      success: true,
      transfer: {
        status: 'success',
        txHash: transferResult.txHash,
        to,
        amount: amount.toString(),
        chain,
        method: transferResult.method,
        proofHash,
        proofVerified,
        attestationTxHash,
        crossmintTransactionId: transferResult.crossmintTransactionId,
        integrationMode: INTEGRATION_MODE,
        note: proofVerified
          ? 'Transfer executed after cryptographic proof verification.'
          : 'Transfer executed (proof verification skipped).',
      },
      steps,
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
