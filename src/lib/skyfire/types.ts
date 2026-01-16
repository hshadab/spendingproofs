/**
 * Skyfire Integration Types
 *
 * Type definitions for Skyfire KYA (Know Your Agent) and payment integration.
 */

/**
 * Skyfire agent identity
 */
export interface SkyfireAgent {
  id: string;
  name: string;
  walletAddress: string;
  createdAt: string;
  kyaCredentials: SkyfireKYACredentials;
}

/**
 * KYA (Know Your Agent) credentials
 */
export interface SkyfireKYACredentials {
  agentId: string;
  issuer: string;
  issuedAt: number;
  expiresAt: number;
  capabilities: string[];
  verificationStatus: 'pending' | 'verified' | 'expired';
}

/**
 * KYA Token for identity verification
 */
export interface SkyfireKYAToken {
  token: string;
  agentId: string;
  expiresAt: number;
  claims: {
    sub: string;      // Agent ID
    iss: string;      // Issuer (Skyfire)
    iat: number;      // Issued at
    exp: number;      // Expiry
    capabilities: string[];
  };
}

/**
 * PAY Token for payment authorization
 */
export interface SkyfirePayToken {
  token: string;
  agentId: string;
  amount: number;
  currency: string;
  recipient: string;
  expiresAt: number;
  proofHash?: string;           // Our zkML proof hash
  verificationHash?: string;    // Our verification attestation hash
}

/**
 * Skyfire payment request
 */
export interface SkyfirePaymentRequest {
  agentId: string;
  amount: number;
  currency: string;
  recipient: string;
  proofHash?: string;
  verificationHash?: string;
  memo?: string;
}

/**
 * Skyfire payment result
 */
export interface SkyfirePaymentResult {
  success: boolean;
  transactionId?: string;
  txHash?: string;
  status: 'pending' | 'completed' | 'failed';
  error?: string;
}

/**
 * Skyfire transfer with zkML verification
 */
export interface SkyfireVerifiedTransfer {
  // Skyfire data
  agentId: string;
  kyaToken: string;
  payToken: string;

  // Transfer details
  recipient: string;
  amount: number;
  currency: string;

  // zkML verification data
  proofHash: string;
  verificationHash: string;
  decision: boolean;
  confidence: number;

  // On-chain data
  attestationTxHash?: string;
  transferTxHash?: string;

  // Timestamps
  proofGeneratedAt: number;
  attestedAt?: number;
  transferredAt?: number;
}

/**
 * Skyfire API configuration
 */
export interface SkyfireConfig {
  apiUrl: string;
  mcpUrl: string;
  environment: 'sandbox' | 'production';
  enabled: boolean;
}

/**
 * Walkthrough step definition
 */
export interface SkyfireWalkthroughStep {
  id: string;
  phase: 'intro' | 'identity' | 'proof' | 'attestation' | 'payment' | 'confirmation';
  title: string;
  description: string;
  duration: number;  // milliseconds
  annotation?: {
    title: string;
    subtitle: string;
    metric?: string;
    category: 'skyfire' | 'zkml' | 'combined' | 'enterprise';
  };
}

/**
 * Demo state for walkthrough
 */
export interface SkyfireDemoState {
  mode: 'demo' | 'live';
  currentStep: number;
  isPlaying: boolean;
  agent: SkyfireAgent | null;
  kyaToken: SkyfireKYAToken | null;
  payToken: SkyfirePayToken | null;
  proofHash: string | null;
  verificationHash: string | null;
  attestationTxHash: string | null;
  transferTxHash: string | null;
  error: string | null;
}
