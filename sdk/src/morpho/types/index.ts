/**
 * Core types for Morpho Spending Proofs SDK
 */

export type Address = `0x${string}`;
export type Hex = `0x${string}`;

/**
 * Operation types for Morpho vault operations
 */
export enum MorphoOperation {
  SUPPLY = 0,
  BORROW = 1,
  WITHDRAW = 2,
  REPAY = 3,
}

/**
 * Spending policy configuration
 */
export interface SpendingPolicy {
  /** Maximum daily spend in base units (e.g., USDC with 6 decimals) */
  dailyLimit: bigint;
  /** Maximum single transaction amount */
  maxSingleTx: bigint;
  /** Maximum loan-to-value ratio in basis points (e.g., 7000 = 70%) */
  maxLTV: number;
  /** Minimum health factor in basis points (e.g., 12000 = 1.2) */
  minHealthFactor: number;
  /** Whitelisted Morpho market addresses */
  allowedMarkets: Address[];
  /** Require proof for supply operations */
  requireProofForSupply: boolean;
  /** Require proof for borrow operations */
  requireProofForBorrow: boolean;
  /** Require proof for withdraw operations */
  requireProofForWithdraw: boolean;
}

/**
 * Proof request for Morpho operations
 */
export interface MorphoProofRequest {
  /** The policy to prove against */
  policy: SpendingPolicy;
  /** Type of operation */
  operation: MorphoOperation;
  /** Amount in base units */
  amount: bigint;
  /** Target market address */
  market: Address;
  /** Agent address executing the operation */
  agent: Address;
  /** Current position state for LTV calculation */
  positionState?: PositionState;
}

/**
 * Current position state for risk calculations
 */
export interface PositionState {
  /** Current supply value in USD */
  supplyValueUSD: bigint;
  /** Current borrow value in USD */
  borrowValueUSD: bigint;
  /** Current collateral value in USD */
  collateralValueUSD: bigint;
  /** Current health factor in basis points */
  healthFactor: number;
}

/**
 * Generated spending proof
 */
export interface SpendingProof {
  /** Hash of the policy being proven against */
  policyHash: Hex;
  /** Jolt-Atlas SNARK proof bytes (~48KB) */
  proof: Hex;
  /** Public inputs to the proof */
  publicInputs: Hex[];
  /** Proof generation timestamp */
  timestamp: number;
  /** Agent signature over proof commitment */
  signature: Hex;
}

/**
 * Result of proof verification
 */
export interface ProofVerificationResult {
  /** Whether the proof is valid */
  valid: boolean;
  /** Error message if invalid */
  error?: string;
  /** Gas estimate for on-chain verification */
  gasEstimate?: bigint;
}

/**
 * Agent configuration
 */
export interface AgentConfig {
  /** Agent address */
  agent: Address;
  /** Vault owner who authorized the agent */
  owner: Address;
  /** Active policy hash */
  policyHash: Hex;
  /** Amount spent today */
  dailySpent: bigint;
  /** Last daily reset timestamp */
  lastResetTimestamp: number;
  /** Whether agent is currently active */
  isActive: boolean;
}

/**
 * Morpho market parameters
 */
export interface MarketParams {
  loanToken: Address;
  collateralToken: Address;
  oracle: Address;
  irm: Address;
  lltv: bigint;
}

/**
 * SDK configuration
 */
export interface MorphoSpendingProofsConfig {
  /** Jolt-Atlas prover service URL */
  proverUrl: string;
  /** Chain ID */
  chainId: number;
  /** MorphoSpendingGate contract address */
  gateAddress: Address;
  /** Morpho Blue contract address */
  morphoAddress: Address;
  /** Optional API key for prover service */
  apiKey?: string;
  /** Request timeout in ms */
  timeout?: number;
}

/**
 * Transaction request for gated operations
 */
export interface GatedTxRequest {
  operation: MorphoOperation;
  market: Address;
  amount: bigint;
  onBehalf: Address;
  receiver?: Address;
}

/**
 * Prepared transaction with proof
 */
export interface PreparedGatedTx {
  /** Transaction data to submit */
  data: Hex;
  /** Target contract (MorphoSpendingGate) */
  to: Address;
  /** Value to send (usually 0) */
  value: bigint;
  /** The generated proof */
  proof: SpendingProof;
  /** Gas estimate */
  gasEstimate: bigint;
}
