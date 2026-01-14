'use client';

import { useCallback, useState } from 'react';
import { useSignTypedData, useChainId } from 'wagmi';
import { ARC_CHAIN } from '@/lib/config';

// EIP-712 Domain for ACK Credentials
const ACK_DOMAIN = {
  name: 'Agent Commerce Kit',
  version: '1',
  chainId: ARC_CHAIN.id,
} as const;

// EIP-712 Types for Controller Credential
const CONTROLLER_CREDENTIAL_TYPES = {
  ControllerCredential: [
    { name: 'agentDid', type: 'string' },
    { name: 'agentName', type: 'string' },
    { name: 'controller', type: 'address' },
    { name: 'issuanceDate', type: 'string' },
    { name: 'expirationDate', type: 'string' },
  ],
} as const;

// EIP-712 Types for Payment Receipt Credential
const PAYMENT_RECEIPT_TYPES = {
  PaymentReceipt: [
    { name: 'agentDid', type: 'string' },
    { name: 'txHash', type: 'bytes32' },
    { name: 'amount', type: 'string' },
    { name: 'recipient', type: 'address' },
    { name: 'proofHash', type: 'bytes32' },
    { name: 'issuanceDate', type: 'string' },
  ],
} as const;

export interface ControllerCredentialMessage {
  agentDid: string;
  agentName: string;
  controller: `0x${string}`;
  issuanceDate: string;
  expirationDate: string;
}

export interface PaymentReceiptMessage {
  agentDid: string;
  txHash: `0x${string}`;
  amount: string;
  recipient: `0x${string}`;
  proofHash: `0x${string}`;
  issuanceDate: string;
}

export interface UseCredentialSigningReturn {
  // Sign controller credential
  signControllerCredential: (message: ControllerCredentialMessage) => void;

  // Sign payment receipt
  signPaymentReceipt: (message: PaymentReceiptMessage) => void;

  // State
  signature: `0x${string}` | undefined;
  isPending: boolean;
  isSuccess: boolean;
  error: Error | null;

  // Reset
  reset: () => void;
}

/**
 * Hook for signing ACK credentials with EIP-712
 */
export function useCredentialSigning(): UseCredentialSigningReturn {
  const chainId = useChainId();

  const {
    signTypedData,
    data: signature,
    isPending,
    isSuccess,
    error,
    reset,
  } = useSignTypedData();

  const signControllerCredential = useCallback(
    (message: ControllerCredentialMessage) => {
      signTypedData({
        domain: {
          ...ACK_DOMAIN,
          chainId: chainId || ARC_CHAIN.id,
        },
        types: CONTROLLER_CREDENTIAL_TYPES,
        primaryType: 'ControllerCredential',
        message,
      });
    },
    [signTypedData, chainId]
  );

  const signPaymentReceipt = useCallback(
    (message: PaymentReceiptMessage) => {
      signTypedData({
        domain: {
          ...ACK_DOMAIN,
          chainId: chainId || ARC_CHAIN.id,
        },
        types: PAYMENT_RECEIPT_TYPES,
        primaryType: 'PaymentReceipt',
        message,
      });
    },
    [signTypedData, chainId]
  );

  return {
    signControllerCredential,
    signPaymentReceipt,
    signature,
    isPending,
    isSuccess,
    error: error || null,
    reset,
  };
}

/**
 * Get the EIP-712 domain for verification
 */
export function getACKDomain(chainId?: number) {
  return {
    ...ACK_DOMAIN,
    chainId: chainId || ARC_CHAIN.id,
  };
}

/**
 * Get the EIP-712 types for controller credential
 */
export function getControllerCredentialTypes() {
  return CONTROLLER_CREDENTIAL_TYPES;
}

/**
 * Get the EIP-712 types for payment receipt
 */
export function getPaymentReceiptTypes() {
  return PAYMENT_RECEIPT_TYPES;
}
