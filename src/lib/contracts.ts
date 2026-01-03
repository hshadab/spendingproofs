/**
 * Contract Addresses and ABIs for Arc Testnet
 */

type Address = `0x${string}`;

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
 * SpendingGateWallet ABI - Real on-chain enforcement
 */
export const SPENDING_GATE_WALLET_ABI = [
  // Read functions
  {
    name: 'getBalance',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'getCurrentNonce',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'getRemainingDailyAllowance',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'isProofUsed',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    outputs: [{ name: '', type: 'bool' }],
  },
  {
    name: 'computeTxIntentHash',
    type: 'function',
    stateMutability: 'view',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'amount', type: 'uint256' },
      { name: '_nonce', type: 'uint256' },
      { name: 'expiry', type: 'uint256' },
    ],
    outputs: [{ name: '', type: 'bytes32' }],
  },
  {
    name: 'dailyLimit',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'maxSingleTransfer',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'agentId',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
  // Write functions
  {
    name: 'deposit',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [{ name: 'amount', type: 'uint256' }],
    outputs: [],
  },
  {
    name: 'gatedTransfer',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'to', type: 'address' },
      { name: 'amount', type: 'uint256' },
      { name: 'proofHash', type: 'bytes32' },
      { name: 'expiry', type: 'uint256' },
    ],
    outputs: [],
  },
  {
    name: 'updateLimits',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: '_dailyLimit', type: 'uint256' },
      { name: '_maxSingleTransfer', type: 'uint256' },
    ],
    outputs: [],
  },
  {
    name: 'emergencyWithdraw',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [{ name: 'to', type: 'address' }],
    outputs: [],
  },
  // Events
  {
    name: 'Deposit',
    type: 'event',
    inputs: [
      { name: 'from', type: 'address', indexed: true },
      { name: 'amount', type: 'uint256', indexed: false },
    ],
  },
  {
    name: 'GatedTransfer',
    type: 'event',
    inputs: [
      { name: 'to', type: 'address', indexed: true },
      { name: 'amount', type: 'uint256', indexed: false },
      { name: 'proofHash', type: 'bytes32', indexed: false },
      { name: 'txIntentHash', type: 'bytes32', indexed: false },
    ],
  },
  {
    name: 'EmergencyWithdraw',
    type: 'event',
    inputs: [
      { name: 'to', type: 'address', indexed: true },
      { name: 'amount', type: 'uint256', indexed: false },
    ],
  },
  {
    name: 'LimitsUpdated',
    type: 'event',
    inputs: [
      { name: 'dailyLimit', type: 'uint256', indexed: false },
      { name: 'maxSingleTransfer', type: 'uint256', indexed: false },
    ],
  },
  // Errors (for decoding)
  { name: 'InvalidProof', type: 'error', inputs: [] },
  { name: 'ProofAlreadyUsed', type: 'error', inputs: [] },
  { name: 'ProofNotAttested', type: 'error', inputs: [] },
  { name: 'TxIntentMismatch', type: 'error', inputs: [] },
  { name: 'ProofExpired', type: 'error', inputs: [] },
  { name: 'ExceedsDailyLimit', type: 'error', inputs: [] },
  { name: 'ExceedsMaxSingleTransfer', type: 'error', inputs: [] },
  { name: 'InsufficientBalance', type: 'error', inputs: [] },
  { name: 'InvalidRecipient', type: 'error', inputs: [] },
  { name: 'TransferFailed', type: 'error', inputs: [] },
] as const;

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
