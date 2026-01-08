/**
 * OpenMind Robot Spending Proofs - Type Definitions
 *
 * Types for autonomous robot payments via x402 protocol
 */

/**
 * Robot embodiment types supported by OM1
 */
export type RobotType = 'humanoid' | 'quadruped' | 'delivery' | 'inspection' | 'home_assistant' | 'digital_agent';

/**
 * Service categories robots can pay for
 */
export type ServiceCategory =
  | 'charging'      // Electric charging stations
  | 'navigation'    // Navigation/mapping APIs
  | 'compute'       // Cloud compute resources
  | 'data'          // Sensor data, weather, etc.
  | 'transport'     // Ride services (Waymo, etc.)
  | 'maintenance'   // Repair/maintenance services
  | 'storage'       // Data storage
  | 'communication' // Network/comms services
  | 'other';

/**
 * Spending policy for a robot
 */
export interface RobotSpendingPolicy {
  /** Maximum daily spend in USDC */
  dailyLimitUsdc: number;
  /** Maximum single transaction in USDC */
  maxSingleTxUsdc: number;
  /** Allowed service categories */
  allowedCategories: ServiceCategory[];
  /** Minimum service reliability score (0-1) */
  minServiceReliability: number;
  /** Whether to require proof for all transactions */
  requireProofForAll: boolean;
  /** Emergency override threshold (allows overspend if critical) */
  emergencyThresholdUsdc?: number;
}

/**
 * Robot wallet state
 */
export interface RobotWalletState {
  /** Wallet address */
  address: `0x${string}`;
  /** Current USDC balance */
  balanceUsdc: number;
  /** Amount spent today */
  spentTodayUsdc: number;
  /** Number of transactions today */
  txCountToday: number;
  /** Last transaction timestamp */
  lastTxTimestamp?: number;
}

/**
 * Service being purchased
 */
export interface ServicePayment {
  /** Service provider address or identifier */
  serviceId: string;
  /** Human-readable service name */
  serviceName: string;
  /** Service category */
  category: ServiceCategory;
  /** Price in USDC */
  priceUsdc: number;
  /** Service reliability score (0-1) */
  reliabilityScore: number;
  /** Description of what's being purchased */
  description: string;
}

/**
 * x402 payment request
 */
export interface X402PaymentRequest {
  /** Robot making the payment */
  robotId: string;
  /** Robot type */
  robotType: RobotType;
  /** Service being paid for */
  service: ServicePayment;
  /** Robot's current wallet state */
  walletState: RobotWalletState;
  /** Policy to evaluate against */
  policy: RobotSpendingPolicy;
  /** Timestamp of request */
  timestamp: number;
}

/**
 * Proof input for zkML verification
 */
export interface RobotProofInput {
  // Transaction details
  priceUsdc: number;
  serviceCategory: ServiceCategory;
  serviceReliability: number;

  // Robot financial state
  budgetUsdc: number;
  spentTodayUsdc: number;
  dailyLimitUsdc: number;
  maxSingleTxUsdc: number;

  // Policy flags
  categoryAllowed: boolean;
  reliabilityMet: boolean;
}

/**
 * Proof result from zkML prover
 */
export interface RobotProofResult {
  /** The generated proof */
  proof: string;
  /** Hash of the proof */
  proofHash: string;
  /** Whether payment was approved */
  approved: boolean;
  /** Confidence score */
  confidence: number;
  /** Risk assessment */
  riskScore: number;
  /** Generation time in ms */
  generationTimeMs: number;
  /** Proof size in bytes */
  proofSizeBytes: number;
  /** Proof metadata */
  metadata: {
    modelHash: string;
    inputHash: string;
    outputHash: string;
    robotId: string;
    serviceId: string;
  };
}

/**
 * Demo step for guided walkthrough
 */
export interface DemoStep {
  id: string;
  phase: 'intro' | 'robot' | 'service' | 'policy' | 'proof' | 'payment' | 'conclusion';
  title: string;
  description: string;
  technicalNote?: string;
  duration: number;
}

/**
 * Robot status for demo visualization
 */
export interface RobotStatus {
  id: string;
  name: string;
  type: RobotType;
  status: 'idle' | 'evaluating' | 'proving' | 'paying' | 'complete' | 'rejected';
  batteryLevel?: number;
  location?: string;
  currentTask?: string;
}
