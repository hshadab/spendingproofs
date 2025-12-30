/**
 * Arc Policy Proofs SDK - Wallet Integration
 *
 * High-level wallet class for spending proofs with automatic
 * proof generation and transaction execution.
 */

import { PolicyProofs } from './client';
import { SpendingInput, ProofResult, ARC_TESTNET } from './types';
import { hashInputs, spendingInputToArray } from './utils';

/**
 * Transaction intent for proof binding
 */
export interface TxIntent {
  chainId: number;
  usdcAddress: string;
  sender: string;
  recipient: string;
  amount: bigint;
  nonce: bigint;
  expiry: number;
  policyId: string;
  policyVersion: number;
}

/**
 * Wallet configuration
 */
export interface SpendingWalletConfig {
  /** Prover service URL */
  proverUrl: string;
  /** Chain ID (default: Arc Testnet) */
  chainId?: number;
  /** Agent address */
  agentAddress: string;
  /** USDC contract address */
  usdcAddress?: string;
  /** Default policy ID */
  policyId?: string;
  /** Policy version */
  policyVersion?: number;
}

/**
 * Gated transfer parameters
 */
export interface GatedTransferParams {
  recipient: string;
  amountUsdc: number;
  input: SpendingInput;
}

/**
 * Gated transfer result
 */
export interface GatedTransferResult {
  proof: ProofResult;
  txIntent: TxIntent;
  txIntentHash: string;
  approved: boolean;
}

/**
 * SpendingProofsWallet - High-level wallet for proof-gated spending
 *
 * @example
 * ```typescript
 * import { SpendingProofsWallet } from '@icme-labs/spending-proofs/wallet';
 *
 * const wallet = new SpendingProofsWallet({
 *   proverUrl: 'http://localhost:3001',
 *   agentAddress: '0x...',
 * });
 *
 * // Prepare a gated transfer
 * const result = await wallet.prepareGatedTransfer({
 *   recipient: '0x...',
 *   amountUsdc: 50,
 *   input: {
 *     priceUsdc: 50,
 *     budgetUsdc: 100,
 *     // ... other inputs
 *   },
 * });
 *
 * if (result.approved) {
 *   // Execute the transfer with your wallet
 *   await executeTransfer(result);
 * }
 * ```
 */
export class SpendingProofsWallet {
  private readonly client: PolicyProofs;
  private readonly chainId: number;
  private readonly agentAddress: string;
  private readonly usdcAddress: string;
  private readonly policyId: string;
  private readonly policyVersion: number;
  private nonce: bigint;

  constructor(config: SpendingWalletConfig) {
    this.client = new PolicyProofs({ proverUrl: config.proverUrl });
    this.chainId = config.chainId ?? ARC_TESTNET.chainId;
    this.agentAddress = config.agentAddress;
    this.usdcAddress = config.usdcAddress ?? '0x0000000000000000000000000000000000000000';
    this.policyId = config.policyId ?? 'default-spending-policy';
    this.policyVersion = config.policyVersion ?? 1;
    this.nonce = BigInt(Date.now());
  }

  /**
   * Generate a unique nonce for a transaction
   */
  private generateNonce(): bigint {
    this.nonce = BigInt(Date.now()) * BigInt(1000) + BigInt(Math.floor(Math.random() * 1000));
    return this.nonce;
  }

  /**
   * Compute transaction intent hash
   */
  computeTxIntentHash(intent: TxIntent): string {
    // Simplified hash for demo - in production use keccak256 with encodePacked
    const parts = [
      intent.chainId.toString(),
      intent.usdcAddress,
      intent.sender,
      intent.recipient,
      intent.amount.toString(),
      intent.nonce.toString(),
      intent.expiry.toString(),
      intent.policyId,
      intent.policyVersion.toString(),
    ];
    return hashInputs(parts.map((p) => parseInt(p, 16) || 0).slice(0, 8));
  }

  /**
   * Create a transaction intent
   */
  createTxIntent(params: {
    recipient: string;
    amountUsdc: number;
    expirySeconds?: number;
  }): TxIntent {
    const nonce = this.generateNonce();
    const expiry = Math.floor(Date.now() / 1000) + (params.expirySeconds ?? 3600);

    return {
      chainId: this.chainId,
      usdcAddress: this.usdcAddress,
      sender: this.agentAddress,
      recipient: params.recipient,
      amount: BigInt(Math.floor(params.amountUsdc * 1_000_000)), // USDC has 6 decimals
      nonce,
      expiry,
      policyId: this.policyId,
      policyVersion: this.policyVersion,
    };
  }

  /**
   * Generate proof and prepare for gated transfer
   */
  async prepareGatedTransfer(params: GatedTransferParams): Promise<GatedTransferResult> {
    // Generate the proof
    const proof = await this.client.prove(params.input);

    // Create transaction intent
    const txIntent = this.createTxIntent({
      recipient: params.recipient,
      amountUsdc: params.amountUsdc,
    });

    // Compute intent hash
    const txIntentHash = this.computeTxIntentHash(txIntent);

    return {
      proof,
      txIntent,
      txIntentHash,
      approved: proof.decision.shouldBuy,
    };
  }

  /**
   * Quick decision without proof (for UI preview)
   */
  decide(input: SpendingInput) {
    return this.client.decide(input);
  }

  /**
   * Check prover health
   */
  async health() {
    return this.client.health();
  }

  /**
   * Get the underlying PolicyProofs client
   */
  getClient(): PolicyProofs {
    return this.client;
  }

  /**
   * Get current chain configuration
   */
  getChainConfig() {
    return {
      chainId: this.chainId,
      agentAddress: this.agentAddress,
      usdcAddress: this.usdcAddress,
      policyId: this.policyId,
      policyVersion: this.policyVersion,
    };
  }
}
