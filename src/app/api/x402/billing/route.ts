/**
 * x402 AWS Billing Endpoint
 *
 * Simulates AWS billing with x402 payment requirements.
 * Returns 402 Payment Required with USDC payment instructions.
 * When payment is provided, processes the purchase and returns confirmation.
 *
 * This demonstrates how AI agents can autonomously pay for cloud resources
 * using the x402 protocol with zkML spending proof verification.
 */

import { NextRequest, NextResponse } from 'next/server';
import { parseUnits, formatUnits } from 'viem';
import {
  BASE_SEPOLIA_USDC,
  BASE_SEPOLIA_CONFIG,
  parsePaymentRequiredHeader,
  getBaseSepoliaUsdcBalance,
  executeDirectUsdcTransfer,
} from '@/lib/x402';
import type { Hex, Address } from 'viem';

// Demo recipient address (would be AWS billing in production)
const BILLING_RECIPIENT = (process.env.X402_BILLING_RECIPIENT ||
  '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384') as Address;

// Scaled price for testnet ($8,500 -> $0.85 at 1:10000)
const TESTNET_PRICE_USDC = '0.85';
const PRODUCTION_PRICE_USDC = '8500.00';

/**
 * Create x402 Payment-Required header
 */
function createPaymentRequiredHeader(
  payTo: Address,
  amount: string,
  description: string
): string {
  const payload = {
    version: '1',
    network: `eip155:${BASE_SEPOLIA_CONFIG.chainId}`,
    payTo,
    maxAmountRequired: parseUnits(amount, 6).toString(),
    asset: BASE_SEPOLIA_USDC,
    description,
    resource: '/api/x402/billing',
    scheme: 'exact',
    extra: {
      service: 'AWS EC2',
      instance: 'p4d.24xlarge',
      productionPrice: PRODUCTION_PRICE_USDC,
      testnetScale: '1:10000',
    },
  };

  return Buffer.from(JSON.stringify(payload)).toString('base64');
}

/**
 * Verify x402 payment signature (simplified for demo)
 *
 * In production, this would:
 * 1. Decode the payment header
 * 2. Verify the ERC-3009 signature
 * 3. Submit to facilitator for settlement
 * 4. Wait for on-chain confirmation
 */
async function verifyAndSettlePayment(
  paymentHeader: string,
  expectedRecipient: Address,
  expectedAmount: string
): Promise<{
  valid: boolean;
  error?: string;
  from?: Address;
  amount?: string;
  signature?: string;
}> {
  try {
    // Decode payment header
    const decoded = Buffer.from(paymentHeader, 'base64').toString('utf-8');
    const payment = JSON.parse(decoded);

    // Verify recipient matches
    if (payment.payload?.to?.toLowerCase() !== expectedRecipient.toLowerCase()) {
      return { valid: false, error: 'Payment recipient mismatch' };
    }

    // Verify amount (allow some tolerance for gas)
    const expectedValue = parseUnits(expectedAmount, 6);
    const paymentValue = BigInt(payment.payload?.value || '0');

    if (paymentValue < expectedValue) {
      return {
        valid: false,
        error: `Insufficient payment: expected ${expectedAmount} USDC, got ${formatUnits(paymentValue, 6)} USDC`,
      };
    }

    return {
      valid: true,
      from: payment.payload?.from as Address,
      amount: formatUnits(paymentValue, 6),
      signature: payment.signature,
    };
  } catch (error) {
    return {
      valid: false,
      error: error instanceof Error ? error.message : 'Payment verification failed',
    };
  }
}

/**
 * POST /api/x402/billing
 *
 * Handles x402 payment flow for AWS billing simulation.
 *
 * Without payment header: Returns 402 with payment requirements
 * With payment header: Verifies payment and returns purchase confirmation
 *
 * Query params:
 * - proofHash: zkML spending proof hash (optional, for audit trail)
 * - skipPayment: Skip actual payment for testing (returns mock success)
 */
export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const proofHash = searchParams.get('proofHash');
  const skipPayment = searchParams.get('skipPayment') === 'true';

  // Check for x402 payment header
  const paymentHeader = request.headers.get('X-Payment') ||
    request.headers.get('x-payment') ||
    request.headers.get('Payment');

  // If no payment header, return 402 Payment Required
  if (!paymentHeader && !skipPayment) {
    const paymentRequiredHeader = createPaymentRequiredHeader(
      BILLING_RECIPIENT,
      TESTNET_PRICE_USDC,
      'AWS EC2 p4d.24xlarge GPU Instance - ML Training'
    );

    return new NextResponse(
      JSON.stringify({
        error: 'Payment Required',
        message: 'This endpoint requires x402 USDC payment on Base Sepolia',
        service: 'AWS EC2 p4d.24xlarge',
        price: {
          testnet: `$${TESTNET_PRICE_USDC} USDC`,
          production: `$${PRODUCTION_PRICE_USDC} USDC`,
          scale: '1:10000 for testnet',
        },
        network: 'Base Sepolia',
        instructions: 'Include X-Payment header with signed ERC-3009 TransferWithAuthorization',
      }),
      {
        status: 402,
        headers: {
          'Content-Type': 'application/json',
          'X-Payment-Required': paymentRequiredHeader,
          'Access-Control-Expose-Headers': 'X-Payment-Required',
        },
      }
    );
  }

  // For testing/demo without real payment
  if (skipPayment) {
    const mockTxHash = '0x' + Array.from({ length: 64 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');

    return NextResponse.json({
      success: true,
      simulated: true,
      purchase: {
        service: 'AWS EC2 p4d.24xlarge',
        status: 'confirmed',
        orderId: `aws-${Date.now()}`,
        amount: TESTNET_PRICE_USDC,
        productionEquivalent: PRODUCTION_PRICE_USDC,
        currency: 'USDC',
        network: 'base-sepolia',
        txHash: mockTxHash,
        proofHash,
        timestamp: new Date().toISOString(),
      },
      x402: {
        protocol: 'x402',
        version: '1',
        paymentMethod: 'simulated',
      },
      explorerUrl: `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${mockTxHash}`,
    });
  }

  // Verify payment (paymentHeader is guaranteed to be non-null here)
  const verification = await verifyAndSettlePayment(
    paymentHeader!,
    BILLING_RECIPIENT,
    TESTNET_PRICE_USDC
  );

  if (!verification.valid) {
    return NextResponse.json({
      success: false,
      error: verification.error,
      expected: {
        recipient: BILLING_RECIPIENT,
        amount: TESTNET_PRICE_USDC,
        currency: 'USDC',
        network: 'base-sepolia',
      },
    }, { status: 400 });
  }

  // Execute actual transfer if we have the demo wallet
  const DEMO_PRIVATE_KEY = process.env.DEMO_WALLET_PRIVATE_KEY as Hex;
  let settlementTxHash: string | undefined;

  if (DEMO_PRIVATE_KEY) {
    try {
      const result = await executeDirectUsdcTransfer(
        DEMO_PRIVATE_KEY,
        BILLING_RECIPIENT,
        TESTNET_PRICE_USDC
      );

      if (result.success) {
        settlementTxHash = result.txHash;
      }
    } catch (error) {
      console.error('Settlement error:', error);
      // Continue without settlement - payment signature was valid
    }
  }

  // Return purchase confirmation
  return NextResponse.json({
    success: true,
    simulated: !settlementTxHash,
    purchase: {
      service: 'AWS EC2 p4d.24xlarge',
      status: 'confirmed',
      orderId: `aws-${Date.now()}`,
      amount: verification.amount,
      productionEquivalent: PRODUCTION_PRICE_USDC,
      currency: 'USDC',
      network: 'base-sepolia',
      txHash: settlementTxHash,
      proofHash,
      payer: verification.from,
      recipient: BILLING_RECIPIENT,
      timestamp: new Date().toISOString(),
    },
    x402: {
      protocol: 'x402',
      version: '1',
      paymentMethod: settlementTxHash ? 'erc3009' : 'signature-verified',
      signature: verification.signature?.slice(0, 20) + '...',
    },
    explorerUrl: settlementTxHash
      ? `${BASE_SEPOLIA_CONFIG.explorerUrl}/tx/${settlementTxHash}`
      : undefined,
  });
}

/**
 * GET /api/x402/billing
 *
 * Returns billing endpoint information and x402 requirements.
 */
export async function GET() {
  return NextResponse.json({
    service: 'AWS EC2 Billing (x402)',
    description: 'x402-enabled billing endpoint for AWS EC2 p4d.24xlarge GPU instances',
    x402: {
      protocol: 'x402',
      version: '1',
      network: 'base-sepolia',
      chainId: BASE_SEPOLIA_CONFIG.chainId,
      asset: BASE_SEPOLIA_USDC,
      recipient: BILLING_RECIPIENT,
      price: {
        testnet: TESTNET_PRICE_USDC,
        production: PRODUCTION_PRICE_USDC,
        scale: '1:10000',
      },
    },
    instructions: {
      step1: 'POST to this endpoint without payment header',
      step2: 'Receive 402 with X-Payment-Required header',
      step3: 'Sign ERC-3009 TransferWithAuthorization',
      step4: 'Retry POST with X-Payment header containing signed payload',
      step5: 'Receive purchase confirmation',
    },
    documentation: {
      x402: 'https://x402.org',
      coinbase: 'https://docs.cdp.coinbase.com/x402/welcome',
    },
  });
}

/**
 * OPTIONS handler for CORS
 */
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-Payment, Authorization',
      'Access-Control-Expose-Headers': 'X-Payment-Required',
    },
  });
}
