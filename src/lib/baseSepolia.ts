/**
 * Base Sepolia Contract Integration
 *
 * Integration with:
 * - ProofAttestation: Submit and verify zkML proof hashes on Base Sepolia
 */

import { createPublicClient, createWalletClient, http, parseAbi, type Hex } from 'viem';
import { baseSepolia } from 'viem/chains';
import { privateKeyToAccount } from 'viem/accounts';
import { createLogger } from './metrics';

const logger = createLogger('lib:baseSepolia');

// Contract addresses on Base Sepolia
export const BASE_SEPOLIA_CONTRACTS = {
  PROOF_ATTESTATION: (process.env.NEXT_PUBLIC_BASE_SEPOLIA_PROOF_ATTESTATION ||
    '0x1Fb62895099b7931FFaBEa1AdF92e20Df7F29213') as Hex,
  SPENDING_GATE: (process.env.NEXT_PUBLIC_BASE_SEPOLIA_SPENDING_GATE ||
    '0x6A47D13593c00359a1c5Fc6f9716926aF184d138') as Hex,
  USDC: (process.env.NEXT_PUBLIC_BASE_SEPOLIA_USDC ||
    '0x3e4ed2d6d6235f9d26707fd5d5af476fb9c91b0f') as Hex,
};

// ABIs
const PROOF_ATTESTATION_ABI = parseAbi([
  'function attestProof(bytes32 proofHash) external',
  'function isProofHashValid(bytes32 proofHash) external view returns (bool)',
  'function getProofTimestamp(bytes32 proofHash) external view returns (uint256)',
  'function totalProofs() external view returns (uint256)',
  'event ProofAttested(bytes32 indexed proofHash, address indexed attester, uint256 timestamp)',
]);

// Create public client for read operations
export function getPublicClient() {
  return createPublicClient({
    chain: baseSepolia,
    transport: http(),
  });
}

// Create wallet client for write operations
export function getWalletClient(privateKey: Hex) {
  const account = privateKeyToAccount(privateKey);
  return createWalletClient({
    account,
    chain: baseSepolia,
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
      address: BASE_SEPOLIA_CONTRACTS.PROOF_ATTESTATION,
      abi: PROOF_ATTESTATION_ABI,
      functionName: 'isProofHashValid',
      args: [proofHash],
    });
    return isValid as boolean;
  } catch (error) {
    logger.error('Error checking proof attestation on Base Sepolia', { action: 'check_attestation', error });
    return false;
  }
}

/**
 * Submit a proof attestation to Base Sepolia
 */
export async function submitProofAttestation(
  privateKey: Hex,
  proofHash: Hex
): Promise<{ success: boolean; txHash?: string; error?: string }> {
  try {
    const walletClient = getWalletClient(privateKey);
    const publicClient = getPublicClient();

    logger.info('Submitting proof attestation to Base Sepolia', {
      action: 'submit_attestation',
      proofHash,
      contract: BASE_SEPOLIA_CONTRACTS.PROOF_ATTESTATION,
    });

    // Submit the proof attestation
    const hash = await walletClient.writeContract({
      address: BASE_SEPOLIA_CONTRACTS.PROOF_ATTESTATION,
      abi: PROOF_ATTESTATION_ABI,
      functionName: 'attestProof',
      args: [proofHash],
    });

    logger.info('Proof attestation submitted', { action: 'submit_attestation', txHash: hash });

    // Wait for confirmation
    const receipt = await publicClient.waitForTransactionReceipt({ hash });

    if (receipt.status === 'success') {
      return { success: true, txHash: hash };
    } else {
      return { success: false, error: 'Transaction reverted' };
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Failed to submit proof attestation', { action: 'submit_attestation', error: errorMessage });
    return { success: false, error: errorMessage };
  }
}

/**
 * Format proof hash to bytes32
 */
export function formatProofHash(proofHash: string): Hex {
  // If already 0x prefixed and 66 chars (32 bytes), return as-is
  if (proofHash.startsWith('0x') && proofHash.length === 66) {
    return proofHash as Hex;
  }
  // If not 0x prefixed, add it
  if (!proofHash.startsWith('0x')) {
    return `0x${proofHash}` as Hex;
  }
  // If too short, pad with zeros
  if (proofHash.length < 66) {
    return `0x${proofHash.slice(2).padStart(64, '0')}` as Hex;
  }
  // If too long, truncate
  return `0x${proofHash.slice(2, 66)}` as Hex;
}

/**
 * Get the explorer URL for a transaction
 */
export function getExplorerTxUrl(txHash: string): string {
  return `https://sepolia.basescan.org/tx/${txHash}`;
}

/**
 * Get the explorer URL for an address
 */
export function getExplorerAddressUrl(address: string): string {
  return `https://sepolia.basescan.org/address/${address}`;
}
