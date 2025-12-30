'use client';

import { useWriteContract, useWaitForTransactionReceipt, useReadContract } from 'wagmi';
import { keccak256, toBytes, encodeAbiParameters } from 'viem';
import { CONTRACTS, getExplorerUrl } from '@/lib/wagmi';
import { useCallback, useMemo } from 'react';

// ArcProofAttestation ABI (subset for our needs)
const PROOF_ATTESTATION_ABI = [
  {
    name: 'isProofValid',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'isProofHashValid',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'proofHash', type: 'bytes32' }],
    outputs: [{ type: 'bool' }],
  },
  {
    name: 'getValidationStatus',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [
      { name: 'validator', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'response', type: 'bool' },
      { name: 'tag', type: 'bytes32' },
      { name: 'timestamp', type: 'uint256' },
    ],
  },
  {
    name: 'getProofMetadata',
    type: 'function',
    stateMutability: 'view',
    inputs: [{ name: 'requestHash', type: 'bytes32' }],
    outputs: [
      { name: 'modelHash', type: 'bytes32' },
      { name: 'inputHash', type: 'bytes32' },
      { name: 'outputHash', type: 'bytes32' },
      { name: 'proofSize', type: 'uint256' },
      { name: 'generationTime', type: 'uint256' },
      { name: 'proverVersion', type: 'string' },
    ],
  },
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
  {
    name: 'validationRequestWithMetadata',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: 'validator', type: 'address' },
      { name: 'agentId', type: 'uint256' },
      { name: 'requestUri', type: 'string' },
      { name: 'proofHash', type: 'bytes32' },
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
] as const;

export interface AttestationParams {
  validatorAddress: `0x${string}`;
  agentId: bigint;
  requestUri: string;
  proofHash: `0x${string}`;
  tag?: string;
  metadata?: {
    modelHash: `0x${string}`;
    inputHash: `0x${string}`;
    outputHash: `0x${string}`;
    proofSize: bigint;
    generationTime: bigint;
    proverVersion: string;
  };
}

export function useAttestation() {
  const attestationAddress = CONTRACTS.proofAttestation;

  // Write contract
  const {
    writeContract,
    data: hash,
    isPending,
    error: writeError,
    reset: resetWrite,
  } = useWriteContract();

  // Wait for receipt
  const {
    isLoading: isConfirming,
    isSuccess,
    data: receipt,
    error: receiptError,
  } = useWaitForTransactionReceipt({ hash });

  // Submit attestation
  const submitAttestation = useCallback(
    async (params: AttestationParams) => {
      const tagBytes32 = params.tag
        ? keccak256(toBytes(params.tag))
        : ('0x' + '0'.repeat(64)) as `0x${string}`;

      if (params.metadata) {
        // Full attestation with metadata
        writeContract({
          address: attestationAddress,
          abi: PROOF_ATTESTATION_ABI,
          functionName: 'validationRequestWithMetadata',
          args: [
            params.validatorAddress,
            params.agentId,
            params.requestUri,
            params.proofHash,
            tagBytes32,
            {
              modelHash: params.metadata.modelHash,
              inputHash: params.metadata.inputHash,
              outputHash: params.metadata.outputHash,
              proofSize: params.metadata.proofSize,
              generationTime: params.metadata.generationTime,
              proverVersion: params.metadata.proverVersion,
            },
          ],
        });
      } else {
        // Simple attestation
        writeContract({
          address: attestationAddress,
          abi: PROOF_ATTESTATION_ABI,
          functionName: 'validationRequest',
          args: [
            params.validatorAddress,
            params.agentId,
            params.requestUri,
            params.proofHash,
          ],
        });
      }
    },
    [attestationAddress, writeContract]
  );

  // Explorer URL
  const explorerUrl = useMemo(() => {
    return hash ? getExplorerUrl('tx', hash) : undefined;
  }, [hash]);

  const error = writeError || receiptError;

  return {
    // State
    hash,
    isPending,
    isConfirming,
    isSuccess,
    receipt,
    error,
    explorerUrl,

    // Actions
    submitAttestation,
    reset: resetWrite,

    // Contract address
    contractAddress: attestationAddress,
  };
}

// Hook to check if a proof is already attested
export function useProofAttested(proofHash: `0x${string}` | undefined) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACTS.proofAttestation,
    abi: PROOF_ATTESTATION_ABI,
    functionName: 'isProofHashValid',
    args: proofHash ? [proofHash] : undefined,
    query: {
      enabled: !!proofHash,
    },
  });

  return {
    isAttested: data as boolean | undefined,
    isLoading,
    error,
    refetch,
  };
}

// Hook to get validation status
export function useValidationStatus(requestHash: `0x${string}` | undefined) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: CONTRACTS.proofAttestation,
    abi: PROOF_ATTESTATION_ABI,
    functionName: 'getValidationStatus',
    args: requestHash ? [requestHash] : undefined,
    query: {
      enabled: !!requestHash,
    },
  });

  const status = useMemo(() => {
    if (!data) return undefined;
    const [validator, agentId, response, tag, timestamp] = data as [
      `0x${string}`,
      bigint,
      boolean,
      `0x${string}`,
      bigint
    ];
    return {
      validator,
      agentId,
      response,
      tag,
      timestamp: new Date(Number(timestamp) * 1000),
    };
  }, [data]);

  return {
    status,
    isLoading,
    error,
    refetch,
  };
}

// Helper to compute request hash
export function computeRequestHash(
  validator: `0x${string}`,
  agentId: bigint,
  requestUri: string
): `0x${string}` {
  const encoded = encodeAbiParameters(
    [
      { type: 'address' },
      { type: 'uint256' },
      { type: 'string' },
    ],
    [validator, agentId, requestUri]
  );
  return keccak256(encoded);
}
