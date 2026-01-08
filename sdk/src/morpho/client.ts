/**
 * Morpho Spending Proofs Client
 *
 * Main SDK client for generating zkML proofs for Morpho operations
 */

import type {
  MorphoSpendingProofsConfig,
  SpendingPolicy,
  MorphoProofRequest,
  SpendingProof,
  GatedTxRequest,
  PreparedGatedTx,
  AgentConfig,
  ProofVerificationResult,
  Address,
  Hex,
} from './types';
import { MorphoOperation } from './types';
import { computePolicyHash } from './policies';

/**
 * Proof generation status
 */
export type ProofStatus = 'idle' | 'preparing' | 'generating' | 'signing' | 'complete' | 'error';

/**
 * Proof generation progress callback
 */
export type ProofProgressCallback = (status: ProofStatus, progress: number) => void;

/**
 * MorphoSpendingProofsClient - Main SDK entry point
 */
export class MorphoSpendingProofsClient {
  private readonly config: Required<MorphoSpendingProofsConfig>;
  private readonly cache: Map<string, SpendingProof> = new Map();
  private nonce: bigint = BigInt(0);

  constructor(config: MorphoSpendingProofsConfig) {
    this.config = {
      ...config,
      apiKey: config.apiKey ?? '',
      timeout: config.timeout ?? 30000,
    };
  }

  /**
   * Generate a spending proof for a Morpho operation
   */
  async generateProof(
    request: MorphoProofRequest,
    signer: { signMessage: (message: string) => Promise<string> },
    onProgress?: ProofProgressCallback,
  ): Promise<SpendingProof> {
    onProgress?.('preparing', 0);

    // Check cache first
    const cacheKey = this.getCacheKey(request);
    const cached = this.cache.get(cacheKey);
    if (cached && this.isProofValid(cached)) {
      onProgress?.('complete', 100);
      return cached;
    }

    onProgress?.('generating', 20);

    // Prepare proof request for Jolt-Atlas prover
    const policyHash = computePolicyHash(request.policy);
    const proofRequest = this.buildProofRequest(request, policyHash);

    // Call prover service
    const proofData = await this.callProver(proofRequest, onProgress);

    onProgress?.('signing', 80);

    // Sign the proof
    const proofCommitment = this.computeProofCommitment(proofData, this.nonce);
    const signature = await signer.signMessage(proofCommitment);

    const proof: SpendingProof = {
      policyHash,
      proof: proofData,
      publicInputs: this.buildPublicInputs(request, policyHash),
      timestamp: Math.floor(Date.now() / 1000),
      signature: signature as Hex,
    };

    // Cache the proof
    this.cache.set(cacheKey, proof);
    this.nonce += BigInt(1);

    onProgress?.('complete', 100);
    return proof;
  }

  /**
   * Prepare a gated transaction with proof
   */
  async prepareGatedTransaction(
    request: GatedTxRequest,
    policy: SpendingPolicy,
    signer: { signMessage: (message: string) => Promise<string> },
    onProgress?: ProofProgressCallback,
  ): Promise<PreparedGatedTx> {
    // Generate the proof
    const proof = await this.generateProof(
      {
        policy,
        operation: request.operation,
        amount: request.amount,
        market: request.market,
        agent: request.onBehalf,
      },
      signer,
      onProgress,
    );

    // Encode the transaction data
    const data = this.encodeGatedTx(request, proof);

    return {
      data,
      to: this.config.gateAddress,
      value: BigInt(0),
      proof,
      gasEstimate: this.estimateGas(request.operation),
    };
  }

  /**
   * Verify a proof locally (without on-chain call)
   */
  async verifyProofLocally(proof: SpendingProof): Promise<ProofVerificationResult> {
    // Basic validation
    if (!this.isProofValid(proof)) {
      return { valid: false, error: 'Proof expired' };
    }

    if (proof.proof.length < 512) {
      return { valid: false, error: 'Proof too short' };
    }

    if (proof.publicInputs.length < 7) {
      return { valid: false, error: 'Insufficient public inputs' };
    }

    // In production, this would verify the SNARK proof locally
    // For now, we do basic structural validation
    return {
      valid: true,
      gasEstimate: BigInt(200000),
    };
  }

  /**
   * Get agent configuration from chain
   */
  async getAgentConfig(agent: Address): Promise<AgentConfig | null> {
    // In production, this would call the contract
    // Placeholder implementation
    return null;
  }

  /**
   * Get remaining daily limit for an agent
   */
  async getRemainingDailyLimit(agent: Address): Promise<bigint> {
    // In production, this would call the contract
    return BigInt(0);
  }

  /**
   * Register a policy on-chain
   */
  async registerPolicy(policy: SpendingPolicy): Promise<Hex> {
    const policyHash = computePolicyHash(policy);
    // In production, this would send a transaction
    return policyHash;
  }

  // ============ Private Methods ============

  private getCacheKey(request: MorphoProofRequest): string {
    return `${request.market}-${request.operation}-${request.amount}-${request.agent}`;
  }

  private isProofValid(proof: SpendingProof): boolean {
    const now = Math.floor(Date.now() / 1000);
    const validityWindow = 300; // 5 minutes
    return now <= proof.timestamp + validityWindow;
  }

  private buildProofRequest(request: MorphoProofRequest, policyHash: Hex): object {
    return {
      policy_hash: policyHash,
      operation_type: request.operation,
      amount: request.amount.toString(),
      market: request.market,
      agent: request.agent,
      position_state: request.positionState
        ? {
            supply_value: request.positionState.supplyValueUSD.toString(),
            borrow_value: request.positionState.borrowValueUSD.toString(),
            collateral_value: request.positionState.collateralValueUSD.toString(),
            health_factor: request.positionState.healthFactor,
          }
        : null,
      nonce: this.nonce.toString(),
    };
  }

  private async callProver(
    request: object,
    onProgress?: ProofProgressCallback,
  ): Promise<Hex> {
    const response = await fetch(`${this.config.proverUrl}/prove`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
      },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`Prover error: ${response.statusText}`);
    }

    // Simulate progress during proof generation
    onProgress?.('generating', 40);
    await new Promise((r) => setTimeout(r, 100));
    onProgress?.('generating', 60);

    const result = await response.json();
    return result.proof as Hex;
  }

  private computeProofCommitment(proof: Hex, nonce: bigint): string {
    // In production, use keccak256
    return `0x${proof.slice(2, 66)}${nonce.toString(16).padStart(64, '0')}`;
  }

  private buildPublicInputs(request: MorphoProofRequest, policyHash: Hex): Hex[] {
    return [
      policyHash,
      `0x${request.operation.toString(16).padStart(64, '0')}` as Hex,
      `0x${request.amount.toString(16).padStart(64, '0')}` as Hex,
      `0x${request.market.slice(2).padStart(64, '0')}` as Hex,
      `0x${request.agent.slice(2).padStart(64, '0')}` as Hex,
      `0x${Math.floor(Date.now() / 1000).toString(16).padStart(64, '0')}` as Hex,
      `0x${this.nonce.toString(16).padStart(64, '0')}` as Hex,
    ];
  }

  private encodeGatedTx(request: GatedTxRequest, proof: SpendingProof): Hex {
    // In production, use ethers/viem to encode the function call
    // This is a placeholder that would encode:
    // supplyWithProof / borrowWithProof / withdrawWithProof / repayWithProof
    const functionSelectors: Record<MorphoOperation, string> = {
      [MorphoOperation.SUPPLY]: '0x12345678',
      [MorphoOperation.BORROW]: '0x23456789',
      [MorphoOperation.WITHDRAW]: '0x34567890',
      [MorphoOperation.REPAY]: '0x45678901',
    };

    return `${functionSelectors[request.operation]}...` as Hex;
  }

  private estimateGas(operation: MorphoOperation): bigint {
    // Base gas + proof verification + Morpho operation
    const baseGas = BigInt(21000);
    const proofVerificationGas = BigInt(200000);
    const operationGas: Record<MorphoOperation, bigint> = {
      [MorphoOperation.SUPPLY]: BigInt(150000),
      [MorphoOperation.BORROW]: BigInt(180000),
      [MorphoOperation.WITHDRAW]: BigInt(160000),
      [MorphoOperation.REPAY]: BigInt(140000),
    };

    return baseGas + proofVerificationGas + operationGas[operation];
  }
}

/**
 * Create a new client instance
 */
export function createMorphoSpendingProofsClient(
  config: MorphoSpendingProofsConfig,
): MorphoSpendingProofsClient {
  return new MorphoSpendingProofsClient(config);
}
