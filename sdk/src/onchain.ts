/**
 * Arc Policy Proofs SDK - On-Chain Verification
 */

import { OnChainVerifyOptions, ARC_TESTNET } from './types';

// RPC response type
interface RpcResponse {
  result?: string;
  error?: { message: string };
}

/**
 * ArcProofAttestation contract ABI (minimal)
 */
export const PROOF_ATTESTATION_ABI = [
  {
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    name: 'isProofValid',
    outputs: [{ type: 'bool' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    name: 'getProofTimestamp',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      { name: 'proofHash', type: 'bytes32' },
      { name: 'metadata', type: 'bytes' },
    ],
    name: 'submitProof',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
] as const;

/**
 * Check if a proof has been attested on-chain
 * Note: This checks if the proof hash was recorded, NOT cryptographic verification
 *
 * @param proofHash - The proof hash to check
 * @param options - On-chain options
 * @returns Whether the proof is attested on-chain
 */
export async function isProofAttested(
  proofHash: string,
  options?: Partial<OnChainVerifyOptions>
): Promise<boolean> {
  const contractAddress =
    options?.contractAddress ?? ARC_TESTNET.contracts.proofAttestation;
  const rpcUrl = options?.rpcUrl ?? ARC_TESTNET.rpcUrl;

  // Encode the function call
  const functionSelector = '0x8f742d16'; // keccak256("isProofValid(bytes32)").slice(0, 10)
  const paddedHash = proofHash.replace('0x', '').padStart(64, '0');
  const data = functionSelector + paddedHash;

  const response = await fetch(rpcUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'eth_call',
      params: [
        {
          to: contractAddress,
          data,
        },
        'latest',
      ],
      id: 1,
    }),
  });

  const result = (await response.json()) as RpcResponse;

  if (result.error) {
    throw new Error(`RPC error: ${result.error.message}`);
  }

  // Result is 0x...0001 for true, 0x...0000 for false
  return Boolean(result.result && result.result !== '0x' + '0'.repeat(64));
}

/**
 * Get proof submission timestamp from chain
 *
 * @param proofHash - The proof hash to check
 * @param options - On-chain verification options
 * @returns Timestamp when proof was submitted (0 if not found)
 */
export async function getProofTimestamp(
  proofHash: string,
  options?: Partial<OnChainVerifyOptions>
): Promise<number> {
  const contractAddress =
    options?.contractAddress ?? ARC_TESTNET.contracts.proofAttestation;
  const rpcUrl = options?.rpcUrl ?? ARC_TESTNET.rpcUrl;

  // Encode the function call
  const functionSelector = '0x3b4da69f'; // keccak256("getProofTimestamp(bytes32)").slice(0, 10)
  const paddedHash = proofHash.replace('0x', '').padStart(64, '0');
  const data = functionSelector + paddedHash;

  const response = await fetch(rpcUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'eth_call',
      params: [
        {
          to: contractAddress,
          data,
        },
        'latest',
      ],
      id: 1,
    }),
  });

  const result = (await response.json()) as RpcResponse;

  if (result.error) {
    throw new Error(`RPC error: ${result.error.message}`);
  }

  return parseInt(result.result || '0', 16);
}

/**
 * Encode proof submission transaction data
 *
 * @param proofHash - The proof hash to submit
 * @param metadata - Optional metadata bytes
 * @returns Encoded transaction data
 */
export function encodeSubmitProof(
  proofHash: string,
  metadata: string = '0x'
): string {
  // This is a simplified encoding - use viem's encodeFunctionData in production
  const functionSelector = '0x12345678'; // placeholder
  const paddedHash = proofHash.replace('0x', '').padStart(64, '0');

  return functionSelector + paddedHash + metadata.replace('0x', '');
}

/**
 * Create viem-compatible contract configuration
 */
export function getProofAttestationContract(
  options?: Partial<OnChainVerifyOptions>
) {
  return {
    address:
      options?.contractAddress ?? ARC_TESTNET.contracts.proofAttestation,
    abi: PROOF_ATTESTATION_ABI,
  };
}
