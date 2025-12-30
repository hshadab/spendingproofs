import { NextRequest, NextResponse } from 'next/server';
import { createWalletClient, createPublicClient, http, parseUnits, encodeFunctionData } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';

// Arc Testnet chain config
const arcTestnet = {
  id: 5042002,
  name: 'Arc Testnet',
  nativeCurrency: { name: 'Arc', symbol: 'ARC', decimals: 18 },
  rpcUrls: {
    default: { http: [process.env.NEXT_PUBLIC_ARC_RPC || 'https://rpc.testnet.arc.network'] },
  },
  blockExplorers: {
    default: { name: 'ArcScan', url: 'https://testnet.arcscan.app' },
  },
} as const;

// Contract ABIs
const PROOF_ATTESTATION_ABI = [
  {
    name: 'validationRequest',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'validator', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'requestUri', type: 'string' },
      { name: 'proofHash', type: 'bytes32' },
    ],
    outputs: [],
  },
] as const;

const ERC20_ABI = [
  {
    name: 'transfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'amount', type: 'uint256' },
    ],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'balanceOf',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'account', type: 'address' }],
    outputs: [{ type: 'uint256' }],
  },
] as const;

// Get demo wallet
function getDemoWallet() {
  const privateKey = process.env.DEMO_WALLET_PRIVATE_KEY;
  if (!privateKey) {
    throw new Error('Demo wallet not configured');
  }
  return privateKeyToAccount(privateKey as `0x${string}`);
}

// Create clients
function createClients() {
  const account = getDemoWallet();

  const publicClient = createPublicClient({
    chain: arcTestnet,
    transport: http(),
  });

  const walletClient = createWalletClient({
    account,
    chain: arcTestnet,
    transport: http(),
  });

  return { publicClient, walletClient, account };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, params } = body;

    const { publicClient, walletClient, account } = createClients();

    if (action === 'attestation') {
      // Submit proof attestation
      const { validatorAddress, agentId, requestUri, proofHash } = params;

      const attestationAddress = process.env.NEXT_PUBLIC_PROOF_ATTESTATION as `0x${string}`;

      const hash = await walletClient.writeContract({
        address: attestationAddress,
        abi: PROOF_ATTESTATION_ABI,
        functionName: 'validationRequest',
        args: [validatorAddress, BigInt(agentId), requestUri, proofHash],
      });

      // Wait for confirmation
      const receipt = await publicClient.waitForTransactionReceipt({ hash });

      return NextResponse.json({
        success: true,
        hash,
        receipt: {
          blockNumber: receipt.blockNumber.toString(),
          status: receipt.status,
        },
        explorerUrl: `https://testnet.arcscan.app/tx/${hash}`,
      });
    }

    if (action === 'payment') {
      // Execute USDC payment
      const { to, amount } = params;

      const usdcAddress = process.env.NEXT_PUBLIC_USDC_ADDRESS as `0x${string}`;

      if (!usdcAddress || usdcAddress === '0x0000000000000000000000000000000000000000') {
        return NextResponse.json({
          success: false,
          error: 'USDC not configured',
          simulated: true,
          message: 'Payment simulated (USDC address not configured)',
        });
      }

      const amountWei = parseUnits(amount.toString(), 6); // USDC has 6 decimals

      const hash = await walletClient.writeContract({
        address: usdcAddress,
        abi: ERC20_ABI,
        functionName: 'transfer',
        args: [to, amountWei],
      });

      // Wait for confirmation
      const receipt = await publicClient.waitForTransactionReceipt({ hash });

      return NextResponse.json({
        success: true,
        hash,
        receipt: {
          blockNumber: receipt.blockNumber.toString(),
          status: receipt.status,
        },
        explorerUrl: `https://testnet.arcscan.app/tx/${hash}`,
      });
    }

    if (action === 'balance') {
      // Get demo wallet balance
      const usdcAddress = process.env.NEXT_PUBLIC_USDC_ADDRESS as `0x${string}`;

      // Get native balance
      const nativeBalance = await publicClient.getBalance({ address: account.address });

      let usdcBalance = BigInt(0);
      if (usdcAddress && usdcAddress !== '0x0000000000000000000000000000000000000000') {
        usdcBalance = await publicClient.readContract({
          address: usdcAddress,
          abi: ERC20_ABI,
          functionName: 'balanceOf',
          args: [account.address],
        }) as bigint;
      }

      return NextResponse.json({
        success: true,
        address: account.address,
        nativeBalance: nativeBalance.toString(),
        usdcBalance: usdcBalance.toString(),
        usdcFormatted: (Number(usdcBalance) / 1e6).toFixed(2),
      });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (error) {
    console.error('Demo transaction error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Transaction failed'
      },
      { status: 500 }
    );
  }
}

// GET endpoint to check demo wallet status
export async function GET() {
  try {
    const account = getDemoWallet();
    const { publicClient } = createClients();

    const nativeBalance = await publicClient.getBalance({ address: account.address });

    const usdcAddress = process.env.NEXT_PUBLIC_USDC_ADDRESS as `0x${string}`;
    let usdcBalance = BigInt(0);
    let usdcConfigured = false;

    if (usdcAddress && usdcAddress !== '0x0000000000000000000000000000000000000000') {
      usdcConfigured = true;
      try {
        usdcBalance = await publicClient.readContract({
          address: usdcAddress,
          abi: ERC20_ABI,
          functionName: 'balanceOf',
          args: [account.address],
        }) as bigint;
      } catch {
        // USDC contract might not exist yet
      }
    }

    return NextResponse.json({
      address: account.address,
      nativeBalance: nativeBalance.toString(),
      nativeFormatted: (Number(nativeBalance) / 1e18).toFixed(4),
      usdcBalance: usdcBalance.toString(),
      usdcFormatted: (Number(usdcBalance) / 1e6).toFixed(2),
      usdcConfigured,
      funded: nativeBalance > BigInt(0),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to get wallet status' },
      { status: 500 }
    );
  }
}
