/**
 * Contract Addresses and ABIs for Arc Testnet
 */

import { type Address } from 'viem';

/**
 * Deployed contract addresses on Arc Testnet (Chain ID: 5042002)
 */
export const CONTRACTS = {
  arcProofAttestation: (process.env.NEXT_PUBLIC_PROOF_ATTESTATION ||
    '0xBE9a5DF7C551324CB872584C6E5bF56799787952') as Address,
  arcAgent: (process.env.NEXT_PUBLIC_ARC_AGENT ||
    '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384') as Address,
} as const;

/**
 * ArcProofAttestation ABI (subset for demo)
 */
export const ARC_PROOF_ATTESTATION_ABI = [
  // Read functions
  {
    name: 'isProofValid',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [{ name: '', type: 'bool' }],
  },
  {
    name: 'isProofHashValid',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    outputs: [{ name: '', type: 'bool' }],
  },
  {
    name: 'getValidationStatus',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [
      { name: 'validatorAddress', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'response', type: 'uint8' },
      { name: 'tag', type: 'bytes32' },
      { name: 'lastUpdate', type: 'uint256' },
    ],
  },
  {
    name: 'getProofMetadata',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [
      {
        name: '',
        type: 'tuple',
        components: [
          { name: 'modelHash', type: 'bytes32' },
          { name: 'inputHash', type: 'bytes32' },
          { name: 'outputHash', type: 'bytes32' },
          { name: 'proofSize', type: 'uint256' },
          { name: 'generationTime', type: 'uint256' },
          { name: 'proverVersion', type: 'string' },
        ],
      },
    ],
  },
  // Write functions
  {
    name: 'validationRequest',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'validatorAddress', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'requestUri', type: 'string' },
      { name: 'requestHash', type: 'bytes32' },
    ],
    outputs: [],
  },
  {
    name: 'validationRequestWithMetadata',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'validatorAddress', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'requestUri', type: 'string' },
      { name: 'requestHash', type: 'bytes32' },
      { name: 'tag', type: 'bytes32' },
      {
        name: 'metadata',
        type: 'tuple',
        components: [
          { name: 'modelHash', type: 'bytes32' },
          { name: 'inputHash', type: 'bytes32' },
          { name: 'outputHash', type: 'bytes32' },
          { name: 'proofSize', type: 'uint256' },
          { name: 'generationTime', type: 'uint256' },
          { name: 'proverVersion', type: 'string' },
        ],
      },
    ],
    outputs: [],
  },
  // Events
  {
    name: 'ValidationRequest',
    type: 'event',
    inputs: [
      { name: 'validatorAddress', type: 'address', indexed: true },
      { name: 'agentId', type: 'uint256', indexed: true },
      { name: 'requestUri', type: 'string', indexed: false },
      { name: 'requestHash', type: 'bytes32', indexed: false },
    ],
  },
  {
    name: 'ProofMetadataSet',
    type: 'event',
    inputs: [
      { name: 'requestHash', type: 'bytes32', indexed: true },
      { name: 'modelHash', type: 'bytes32', indexed: false },
      { name: 'inputHash', type: 'bytes32', indexed: false },
      { name: 'outputHash', type: 'bytes32', indexed: false },
    ],
  },
] as const;

/**
 * Response codes from the contract
 */
export const RESPONSE_CODES = {
  PENDING: 0,
  VALID: 1,
  INVALID: 2,
  INCONCLUSIVE: 3,
} as const;

/**
 * Convert string to bytes32
 */
export function stringToBytes32(str: string): `0x${string}` {
  const hex = Buffer.from(str).toString('hex').padEnd(64, '0');
  return `0x${hex}` as `0x${string}`;
}

/**
 * Truncate hash for display
 */
export function truncateHash(hash: string | undefined | null, chars: number = 8): string {
  if (!hash) return '';
  if (hash.length <= chars * 2 + 2) return hash;
  return `${hash.slice(0, chars + 2)}...${hash.slice(-chars)}`;
}
