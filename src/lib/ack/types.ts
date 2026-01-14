/**
 * Agent Commerce Kit (ACK) Type Definitions
 *
 * Types for ACK-ID (agent identity) and ACK-Pay (payment receipts)
 */

import type { SpendingProof } from '../types';

/**
 * W3C Verifiable Credential structure
 */
export interface VerifiableCredential {
  '@context': string[];
  type: string[];
  issuer: string;
  issuanceDate: string;
  expirationDate?: string;
  credentialSubject: Record<string, unknown>;
  proof?: {
    type: string;
    created: string;
    verificationMethod: string;
    proofPurpose: string;
    proofValue: string;
  };
}

/**
 * ACK Agent Identity (ACK-ID)
 */
export interface ACKAgentIdentity {
  /** W3C Decentralized Identifier */
  did: string;
  /** Controller credential proving ownership */
  controllerCredential: VerifiableCredential;
  /** Owner address (human/org controlling the agent) */
  ownerAddress: string;
  /** Agent name/label */
  name: string;
  /** Service endpoints for agent discovery */
  serviceEndpoints: ServiceEndpoint[];
  /** Creation timestamp */
  createdAt: number;
}

/**
 * Service endpoint for agent discovery
 */
export interface ServiceEndpoint {
  id: string;
  type: string;
  serviceEndpoint: string;
}

/**
 * ACK Payment Receipt (ACK-Pay)
 */
export interface ACKPaymentReceipt {
  /** Verifiable credential for the receipt */
  receiptCredential: VerifiableCredential;
  /** On-chain transaction hash */
  txHash: string;
  /** Payment amount in USDC */
  amount: string;
  /** Recipient address */
  recipient: string;
  /** Chain ID */
  chainId: number;
  /** Associated proof hash */
  proofHash: string;
  /** Receipt issuance timestamp */
  issuedAt: number;
}

/**
 * Combined proof bundle with ACK identity and receipt
 */
export interface ACKProofBundle {
  /** Agent identity */
  identity: ACKAgentIdentity;
  /** zkML spending proof */
  proof: SpendingProof;
  /** Payment receipt (populated after transfer) */
  receipt?: ACKPaymentReceipt;
  /** Bundle creation timestamp */
  createdAt: number;
}

/**
 * ACK demo step status
 */
export type ACKStepStatus = 'idle' | 'loading' | 'complete' | 'error';

/**
 * ACK demo state
 */
export interface ACKDemoState {
  /** Current step (1-5) */
  currentStep: number;
  /** Agent identity */
  identity: ACKAgentIdentity | null;
  /** Spending proof */
  proof: SpendingProof | null;
  /** Payment receipt */
  receipt: ACKPaymentReceipt | null;
  /** Step statuses */
  stepStatuses: {
    identity: ACKStepStatus;
    policy: ACKStepStatus;
    proof: ACKStepStatus;
    payment: ACKStepStatus;
    receipt: ACKStepStatus;
  };
  /** Error message */
  error: string | null;
}

/**
 * Initial ACK demo state
 */
export const INITIAL_ACK_STATE: ACKDemoState = {
  currentStep: 1,
  identity: null,
  proof: null,
  receipt: null,
  stepStatuses: {
    identity: 'idle',
    policy: 'idle',
    proof: 'idle',
    payment: 'idle',
    receipt: 'idle',
  },
  error: null,
};
