/**
 * Arc Network Contract Integration
 *
 * Full integration with:
 * - ProofAttestation: Submit and verify zkML proof hashes
 * - SpendingGateWallet: Gated transfers requiring proof verification
 * - USDC: Token transfers on Arc testnet
 */

import { createPublicClient, createWalletClient, http, parseAbi, type Hex } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';

// Arc Testnet configuration
export const ARC_TESTNET = {
  id: 5042002,
  name: 'Arc Testnet',
  nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: ['https://rpc.testnet.arc.network'] } },
  blockExplorers: { default: { name: 'ArcScan', url: 'https://testnet.arcscan.app' } },
} as const;

// Contract addresses
export const CONTRACTS = {
  USDC: process.env.NEXT_PUBLIC_USDC_ADDRESS || '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213',
  PROOF_ATTESTATION: process.env.NEXT_PUBLIC_PROOF_ATTESTATION || '0xBE9a5DF7C551324CB872584C6E5bF56799787952',
  SPENDING_GATE: process.env.NEXT_PUBLIC_SPENDING_GATE_ADDRESS || '0x6A47D13593c00359a1c5Fc6f9716926aF184d138',
};

// ABIs
const PROOF_ATTESTATION_ABI = parseAbi([
  'function isProofValid(bytes32 proofHash) external view returns (bool)',
  'function getProofTimestamp(bytes32 proofHash) external view returns (uint256)',
  'function submitProof(bytes32 proofHash, bytes metadata) external',
  'event ProofSubmitted(bytes32 indexed proofHash, address indexed submitter, uint256 timestamp)',
]);

const SPENDING_GATE_ABI = parseAbi([
  'function gatedTransfer(address to, uint256 amount, bytes32 proofHash, uint256 expiry) external',
  'function getBalance() external view returns (uint256)',
  'function getRemainingDailyAllowance() external view returns (uint256)',
  'function isProofUsed(bytes32 proofHash) external view returns (bool)',
  'function dailyLimit() external view returns (uint256)',
  'function maxSingleTransfer() external view returns (uint256)',
  'function nonce() external view returns (uint256)',
  'function owner() external view returns (address)',
  'event GatedTransfer(address indexed to, uint256 amount, bytes32 proofHash, bytes32 txIntentHash)',
]);

const ERC20_ABI = parseAbi([
  'function balanceOf(address account) external view returns (uint256)',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function approve(address spender, uint256 amount) external returns (bool)',
]);

// Create public client for read operations
export function getPublicClient() {
  return createPublicClient({
    chain: ARC_TESTNET,
    transport: http(),
  });
}

// Create wallet client for write operations
export function getWalletClient(privateKey: Hex) {
  const account = privateKeyToAccount(privateKey);
  return createWalletClient({
    account,
    chain: ARC_TESTNET,
    transport: http(),
  });
}

/**
 * Check if a proof hash is attested on-chain
 */
export async function isProofAttested(proofHash: Hex): Promise<boolean> {
  const client = getPublicClient();
  try {
    const isValid = await client.readContract({
      address: CONTRACTS.PROOF_ATTESTATION as Hex,
      abi: PROOF_ATTESTATION_ABI,
      functionName: 'isProofValid',
      args: [proofHash],
    });
    return isValid as boolean;
  } catch (error) {
    console.error('Error checking proof attestation:', error);
    return false;
  }
}

/**
 * Submit a proof to the ProofAttestation contract
 */
export async function submitProofAttestation(
  privateKey: Hex,
  proofHash: Hex,
  metadata: Hex = '0x'
): Promise<{ success: boolean; txHash?: string; error?: string }> {
  try {
    const client = getWalletClient(privateKey);
    const publicClient = getPublicClient();

    const hash = await client.writeContract({
      address: CONTRACTS.PROOF_ATTESTATION as Hex,
      abi: PROOF_ATTESTATION_ABI,
      functionName: 'submitProof',
      args: [proofHash, metadata],
    });

    // Wait for confirmation
    const receipt = await publicClient.waitForTransactionReceipt({ hash });

    return {
      success: receipt.status === 'success',
      txHash: hash,
    };
  } catch (error) {
    console.error('Proof attestation error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Check if a proof has already been used in SpendingGate
 */
export async function isProofUsed(proofHash: Hex): Promise<boolean> {
  const client = getPublicClient();
  try {
    const isUsed = await client.readContract({
      address: CONTRACTS.SPENDING_GATE as Hex,
      abi: SPENDING_GATE_ABI,
      functionName: 'isProofUsed',
      args: [proofHash],
    });
    return isUsed as boolean;
  } catch (error) {
    console.error('Error checking proof usage:', error);
    return false;
  }
}

/**
 * Get SpendingGate wallet info
 */
export async function getSpendingGateInfo(): Promise<{
  balance: string;
  dailyLimit: string;
  maxSingleTransfer: string;
  remainingDaily: string;
  nonce: number;
  owner: string;
}> {
  const client = getPublicClient();

  try {
    const [balance, dailyLimit, maxSingleTransfer, remainingDaily, nonce, owner] = await Promise.all([
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'getBalance',
      }),
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'dailyLimit',
      }),
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'maxSingleTransfer',
      }),
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'getRemainingDailyAllowance',
      }),
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'nonce',
      }),
      client.readContract({
        address: CONTRACTS.SPENDING_GATE as Hex,
        abi: SPENDING_GATE_ABI,
        functionName: 'owner',
      }),
    ]);

    return {
      balance: (Number(balance) / 1_000_000).toFixed(2),
      dailyLimit: (Number(dailyLimit) / 1_000_000).toFixed(2),
      maxSingleTransfer: (Number(maxSingleTransfer) / 1_000_000).toFixed(2),
      remainingDaily: (Number(remainingDaily) / 1_000_000).toFixed(2),
      nonce: Number(nonce),
      owner: owner as string,
    };
  } catch (error) {
    console.error('Error getting SpendingGate info:', error);
    return {
      balance: '0.00',
      dailyLimit: '1000.00',
      maxSingleTransfer: '100.00',
      remainingDaily: '1000.00',
      nonce: 0,
      owner: '',
    };
  }
}

/**
 * Execute a gated transfer through SpendingGateWallet
 * This is the REAL flow: proof must be attested first
 */
export async function executeGatedTransfer(
  privateKey: Hex,
  to: Hex,
  amountUsdc: number,
  proofHash: Hex,
  expirySeconds: number = 3600
): Promise<{ success: boolean; txHash?: string; error?: string }> {
  try {
    const client = getWalletClient(privateKey);
    const publicClient = getPublicClient();

    // Convert amount to USDC decimals (6)
    const amountWei = BigInt(Math.round(amountUsdc * 1_000_000));
    const expiry = BigInt(Math.floor(Date.now() / 1000) + expirySeconds);

    // Execute gated transfer
    const hash = await client.writeContract({
      address: CONTRACTS.SPENDING_GATE as Hex,
      abi: SPENDING_GATE_ABI,
      functionName: 'gatedTransfer',
      args: [to, amountWei, proofHash, expiry],
    });

    // Wait for confirmation
    const receipt = await publicClient.waitForTransactionReceipt({ hash });

    return {
      success: receipt.status === 'success',
      txHash: hash,
    };
  } catch (error) {
    console.error('Gated transfer error:', error);
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';

    // Parse common revert reasons
    let friendlyError = errorMsg;
    if (errorMsg.includes('ProofNotAttested')) {
      friendlyError = 'Proof not attested on-chain. Submit proof to ProofAttestation first.';
    } else if (errorMsg.includes('ProofAlreadyUsed')) {
      friendlyError = 'This proof has already been used for a transfer.';
    } else if (errorMsg.includes('ExceedsDailyLimit')) {
      friendlyError = 'Transfer exceeds daily spending limit.';
    } else if (errorMsg.includes('ExceedsMaxSingleTransfer')) {
      friendlyError = 'Transfer exceeds max single transfer limit.';
    } else if (errorMsg.includes('InsufficientBalance')) {
      friendlyError = 'SpendingGate wallet has insufficient USDC balance.';
    } else if (errorMsg.includes('ProofExpired')) {
      friendlyError = 'Proof has expired.';
    }

    return {
      success: false,
      error: friendlyError,
    };
  }
}

/**
 * Execute the full verified transfer flow:
 * 1. Check if proof is already attested
 * 2. If not, submit proof attestation
 * 3. Execute gated transfer
 */
export async function executeVerifiedTransfer(
  privateKey: Hex,
  to: Hex,
  amountUsdc: number,
  proofHash: Hex,
  proofMetadata: Hex = '0x'
): Promise<{
  success: boolean;
  attestationTxHash?: string;
  transferTxHash?: string;
  steps: { step: string; status: 'success' | 'failed' | 'skipped'; txHash?: string }[];
  error?: string;
}> {
  const steps: { step: string; status: 'success' | 'failed' | 'skipped'; txHash?: string }[] = [];

  try {
    // Step 1: Check if proof is already attested
    const alreadyAttested = await isProofAttested(proofHash);

    let attestationTxHash: string | undefined;

    if (alreadyAttested) {
      steps.push({ step: 'Proof Attestation', status: 'skipped' });
    } else {
      // Step 2: Submit proof attestation
      const attestResult = await submitProofAttestation(privateKey, proofHash, proofMetadata);

      if (!attestResult.success) {
        steps.push({ step: 'Proof Attestation', status: 'failed' });
        return {
          success: false,
          steps,
          error: `Failed to attest proof: ${attestResult.error}`,
        };
      }

      attestationTxHash = attestResult.txHash;
      steps.push({ step: 'Proof Attestation', status: 'success', txHash: attestResult.txHash });

      // Small delay to ensure attestation is indexed
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Step 3: Check proof not already used
    const proofUsed = await isProofUsed(proofHash);
    if (proofUsed) {
      steps.push({ step: 'Proof Validation', status: 'failed' });
      return {
        success: false,
        attestationTxHash,
        steps,
        error: 'This proof has already been used for a transfer.',
      };
    }
    steps.push({ step: 'Proof Validation', status: 'success' });

    // Step 4: Execute gated transfer
    const transferResult = await executeGatedTransfer(privateKey, to, amountUsdc, proofHash);

    if (!transferResult.success) {
      steps.push({ step: 'Gated Transfer', status: 'failed' });
      return {
        success: false,
        attestationTxHash,
        steps,
        error: transferResult.error,
      };
    }

    steps.push({ step: 'Gated Transfer', status: 'success', txHash: transferResult.txHash });

    return {
      success: true,
      attestationTxHash,
      transferTxHash: transferResult.txHash,
      steps,
    };
  } catch (error) {
    console.error('Verified transfer error:', error);
    return {
      success: false,
      steps,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Execute direct USDC transfer (fallback when SpendingGate not available)
 */
export async function executeDirectTransfer(
  privateKey: Hex,
  to: Hex,
  amountUsdc: number
): Promise<{ success: boolean; txHash?: string; error?: string }> {
  try {
    const client = getWalletClient(privateKey);

    const amountWei = BigInt(Math.round(amountUsdc * 1_000_000));

    const hash = await client.writeContract({
      address: CONTRACTS.USDC as Hex,
      abi: ERC20_ABI,
      functionName: 'transfer',
      args: [to, amountWei],
    });

    return {
      success: true,
      txHash: hash,
    };
  } catch (error) {
    console.error('Direct transfer error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Get USDC balance for an address
 */
export async function getUsdcBalance(address: Hex): Promise<string> {
  const client = getPublicClient();
  try {
    const balance = await client.readContract({
      address: CONTRACTS.USDC as Hex,
      abi: ERC20_ABI,
      functionName: 'balanceOf',
      args: [address],
    });
    return (Number(balance) / 1_000_000).toFixed(2);
  } catch (error) {
    console.error('Error getting USDC balance:', error);
    return '0.00';
  }
}

/**
 * Format proof hash for contract interaction
 */
export function formatProofHash(proofHash: string): Hex {
  // Ensure it's a valid bytes32
  let hash = proofHash;
  if (!hash.startsWith('0x')) {
    hash = '0x' + hash;
  }
  // Pad to 32 bytes if needed
  if (hash.length < 66) {
    hash = '0x' + hash.slice(2).padStart(64, '0');
  }
  return hash as Hex;
}

/**
 * Get Arc Explorer URL for a transaction
 */
export function getExplorerUrl(txHash: string): string {
  return `https://testnet.arcscan.app/tx/${txHash}`;
}

/**
 * Get Arc Explorer URL for a contract
 */
export function getContractExplorerUrl(address: string): string {
  return `https://testnet.arcscan.app/address/${address}`;
}
