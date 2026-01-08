export type Address = `0x${string}`;

export enum MorphoOperation {
  SUPPLY = 0,
  BORROW = 1,
  WITHDRAW = 2,
  REPAY = 3,
}

export interface SpendingPolicy {
  dailyLimit: number;
  maxSingleTx: number;
  maxLTV: number;
  minHealthFactor: number;
  allowedMarkets: string[];
  requireProofForSupply: boolean;
  requireProofForBorrow: boolean;
  requireProofForWithdraw: boolean;
}

export interface AgentConfig {
  address: string;
  ownerAddress: string;
  policyHash: string;
  strategy: string;
  isActive: boolean;
}

export interface MarketData {
  address: string;
  name: string;
  supplyAPY: number;
  borrowAPY: number;
  totalSupply: number;
  totalBorrow: number;
  utilization: number;
}

export interface MorphoMarket {
  address: string;
  name: string;
  loanAsset: string;
  collateralAsset: string;
  supplyAPY: number;
  borrowAPY: number;
  utilization: number;
  totalSupply: number;
  totalBorrow: number;
  lltv: number;
  isLive: boolean;
}

export interface ProofState {
  status: 'idle' | 'preparing' | 'generating' | 'signing' | 'verifying' | 'complete' | 'error';
  progress: number;
  proofSize?: number;
  proofHash?: string;
  gasEstimate?: number;
  error?: string;
}

export interface Transaction {
  id: string;
  timestamp: number;
  operation: MorphoOperation;
  market: string;
  amount: number;
  proofHash: string;
  txHash: string;
  status: 'pending' | 'success' | 'failed';
  gasUsed?: number;
}

export interface AgentDecision {
  operation: MorphoOperation;
  market: string;
  amount: number;
  reasoning: string;
  confidence: number;
}

export interface DemoState {
  step: 'policy' | 'authorize' | 'running' | 'complete';
  policy: SpendingPolicy | null;
  agent: AgentConfig | null;
  transactions: Transaction[];
  dailySpent: number;
  currentProof: ProofState;
  isSimulating: boolean;
}
