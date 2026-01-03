/**
 * Spending Decision Model
 *
 * This model decides WHETHER an agent should purchase a service.
 * zkML proves the agent ran this model correctly before spending.
 *
 * Input: pre-payment data (price, budget, reputation, history)
 * Output: buy/don't_buy decision with confidence
 */

import { ValidationError } from './errors';

export type ServiceCategory =
  | 'ai'
  | 'data'
  | 'compute'
  | 'storage'
  | 'api'
  | 'other';

/**
 * Input to the spending decision model
 * All data available BEFORE payment
 */
export interface SpendingModelInput {
  // Service info (from Bazaar/x402)
  serviceUrl: string;
  serviceName: string;
  serviceCategory: ServiceCategory | string;
  priceUsdc: number;           // Price in USDC (e.g., 0.001)

  // Agent's financial state
  budgetUsdc: number;          // Treasury balance in USDC
  spentTodayUsdc: number;      // Already spent today
  dailyLimitUsdc: number;      // Max daily spend (agent config)

  // Service reputation (tracked locally)
  serviceSuccessRate: number;  // 0-1, historical success rate
  serviceTotalCalls: number;   // How many times we've used this service

  // Agent behavior
  purchasesInCategory: number; // Recent purchases in same category
  timeSinceLastPurchase: number; // Seconds since last purchase (any service)
}

/**
 * Output from the spending decision model
 */
export interface SpendingModelOutput {
  shouldBuy: boolean;
  confidence: number;          // 0-1
  reasons: string[];           // Human-readable explanations
  riskScore: number;           // 0-1, higher = riskier purchase
}

/**
 * Spending model configuration per agent
 */
export interface SpendingPolicy {
  dailyLimitUsdc: number;      // Max spend per day
  maxSinglePurchaseUsdc: number; // Max single transaction
  minSuccessRate: number;      // Won't buy from services below this
  minBudgetBuffer: number;     // Keep at least this much in treasury
  categoryLimits?: Record<string, number>; // Per-category daily limits
}

// Default spending policy
export const DEFAULT_SPENDING_POLICY: SpendingPolicy = {
  dailyLimitUsdc: 1.0,         // $1/day max
  maxSinglePurchaseUsdc: 0.10, // $0.10 max per purchase
  minSuccessRate: 0.5,         // 50% success rate minimum
  minBudgetBuffer: 0.01,       // Keep $0.01 minimum in treasury
};

/**
 * Validation result for spending model input
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Validate spending model input
 * Returns errors array - empty if valid
 */
export function validateSpendingInput(input: SpendingModelInput): ValidationResult {
  const errors: string[] = [];

  // Price validation
  if (typeof input.priceUsdc !== 'number' || isNaN(input.priceUsdc)) {
    errors.push('priceUsdc must be a valid number');
  } else if (input.priceUsdc < 0) {
    errors.push('priceUsdc must be non-negative');
  } else if (input.priceUsdc > 1000000) {
    errors.push('priceUsdc exceeds maximum allowed value (1,000,000)');
  }

  // Budget validation
  if (typeof input.budgetUsdc !== 'number' || isNaN(input.budgetUsdc)) {
    errors.push('budgetUsdc must be a valid number');
  } else if (input.budgetUsdc < 0) {
    errors.push('budgetUsdc must be non-negative');
  }

  // Spent today validation
  if (typeof input.spentTodayUsdc !== 'number' || isNaN(input.spentTodayUsdc)) {
    errors.push('spentTodayUsdc must be a valid number');
  } else if (input.spentTodayUsdc < 0) {
    errors.push('spentTodayUsdc must be non-negative');
  }

  // Daily limit validation
  if (typeof input.dailyLimitUsdc !== 'number' || isNaN(input.dailyLimitUsdc)) {
    errors.push('dailyLimitUsdc must be a valid number');
  } else if (input.dailyLimitUsdc <= 0) {
    errors.push('dailyLimitUsdc must be positive');
  }

  // Service success rate validation (must be 0-1)
  if (typeof input.serviceSuccessRate !== 'number' || isNaN(input.serviceSuccessRate)) {
    errors.push('serviceSuccessRate must be a valid number');
  } else if (input.serviceSuccessRate < 0 || input.serviceSuccessRate > 1) {
    errors.push('serviceSuccessRate must be between 0 and 1');
  }

  // Service total calls validation
  if (typeof input.serviceTotalCalls !== 'number' || isNaN(input.serviceTotalCalls)) {
    errors.push('serviceTotalCalls must be a valid number');
  } else if (input.serviceTotalCalls < 0) {
    errors.push('serviceTotalCalls must be non-negative');
  } else if (!Number.isInteger(input.serviceTotalCalls)) {
    errors.push('serviceTotalCalls must be an integer');
  }

  // Purchases in category validation
  if (typeof input.purchasesInCategory !== 'number' || isNaN(input.purchasesInCategory)) {
    errors.push('purchasesInCategory must be a valid number');
  } else if (input.purchasesInCategory < 0) {
    errors.push('purchasesInCategory must be non-negative');
  } else if (!Number.isInteger(input.purchasesInCategory)) {
    errors.push('purchasesInCategory must be an integer');
  }

  // Time since last purchase validation
  if (typeof input.timeSinceLastPurchase !== 'number' || isNaN(input.timeSinceLastPurchase)) {
    errors.push('timeSinceLastPurchase must be a valid number');
  } else if (input.timeSinceLastPurchase < 0) {
    errors.push('timeSinceLastPurchase must be non-negative');
  }

  // String field validations
  if (typeof input.serviceUrl !== 'string' || input.serviceUrl.trim() === '') {
    errors.push('serviceUrl must be a non-empty string');
  }

  if (typeof input.serviceName !== 'string' || input.serviceName.trim() === '') {
    errors.push('serviceName must be a non-empty string');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate spending model input and throw if invalid
 */
export function assertValidSpendingInput(input: SpendingModelInput): void {
  const result = validateSpendingInput(input);
  if (!result.valid) {
    throw ValidationError.fromErrors(result.errors);
  }
}

export interface RunSpendingModelOptions {
  /** Validate input before running (default: true) */
  validate?: boolean;
}

/**
 * Run the spending decision model
 *
 * This is a deterministic model that can be proven with zkML.
 * The logic is transparent and verifiable.
 */
export function runSpendingModel(
  input: SpendingModelInput,
  policy: SpendingPolicy = DEFAULT_SPENDING_POLICY,
  options: RunSpendingModelOptions = {}
): SpendingModelOutput {
  const { validate = true } = options;

  // Validate input if enabled
  if (validate) {
    assertValidSpendingInput(input);
  }

  const reasons: string[] = [];
  let riskScore = 0;

  // === HARD BLOCKS (shouldBuy = false, no exceptions) ===

  // 1. Price exceeds single purchase limit
  if (input.priceUsdc > policy.maxSinglePurchaseUsdc) {
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: [`Price $${input.priceUsdc.toFixed(4)} exceeds max single purchase $${policy.maxSinglePurchaseUsdc.toFixed(4)}`],
      riskScore: 1.0,
    };
  }

  // 2. Would exceed daily limit
  const projectedDailySpend = input.spentTodayUsdc + input.priceUsdc;
  if (projectedDailySpend > policy.dailyLimitUsdc) {
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: [`Would exceed daily limit: $${projectedDailySpend.toFixed(4)} > $${policy.dailyLimitUsdc.toFixed(4)}`],
      riskScore: 1.0,
    };
  }

  // 3. Insufficient budget (including buffer)
  const availableBudget = input.budgetUsdc - policy.minBudgetBuffer;
  if (input.priceUsdc > availableBudget) {
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: [`Insufficient budget: need $${input.priceUsdc.toFixed(4)}, have $${availableBudget.toFixed(4)} available`],
      riskScore: 1.0,
    };
  }

  // 4. Service has bad reputation (if we have history)
  if (input.serviceTotalCalls >= 3 && input.serviceSuccessRate < policy.minSuccessRate) {
    return {
      shouldBuy: false,
      confidence: 0.9,
      reasons: [`Service success rate ${(input.serviceSuccessRate * 100).toFixed(0)}% below minimum ${(policy.minSuccessRate * 100).toFixed(0)}%`],
      riskScore: 0.9,
    };
  }

  // === SOFT FACTORS (affect confidence and risk score) ===

  // Factor 1: Price relative to budget
  const budgetRatio = input.priceUsdc / input.budgetUsdc;
  if (budgetRatio > 0.5) {
    riskScore += 0.3;
    reasons.push(`High budget ratio: ${(budgetRatio * 100).toFixed(0)}% of treasury`);
  } else if (budgetRatio < 0.1) {
    reasons.push(`Low budget impact: ${(budgetRatio * 100).toFixed(1)}% of treasury`);
  }

  // Factor 2: Service reputation
  if (input.serviceTotalCalls === 0) {
    riskScore += 0.2;
    reasons.push('New service (no history)');
  } else if (input.serviceSuccessRate >= 0.9) {
    riskScore -= 0.1;
    reasons.push(`Trusted service: ${(input.serviceSuccessRate * 100).toFixed(0)}% success rate`);
  } else if (input.serviceSuccessRate < 0.7) {
    riskScore += 0.2;
    reasons.push(`Moderate success rate: ${(input.serviceSuccessRate * 100).toFixed(0)}%`);
  }

  // Factor 3: Daily spend progress
  const dailyProgress = input.spentTodayUsdc / policy.dailyLimitUsdc;
  if (dailyProgress > 0.8) {
    riskScore += 0.15;
    reasons.push(`Near daily limit: ${(dailyProgress * 100).toFixed(0)}% used`);
  }

  // Factor 4: Rapid spending detection
  if (input.timeSinceLastPurchase < 60) { // Less than 1 minute
    riskScore += 0.1;
    reasons.push('Rapid spending detected');
  }

  // Factor 5: Category concentration
  if (input.purchasesInCategory > 5) {
    riskScore += 0.1;
    reasons.push(`High category concentration: ${input.purchasesInCategory} recent purchases`);
  }

  // Normalize risk score
  riskScore = Math.max(0, Math.min(1, riskScore));

  // Calculate confidence (inverse of risk for "buy" decisions)
  const confidence = 1 - (riskScore * 0.5); // Risk affects confidence but doesn't flip decision

  // Decision: buy if risk is acceptable
  const shouldBuy = riskScore < 0.8;

  if (shouldBuy) {
    reasons.unshift(`Approved: price $${input.priceUsdc.toFixed(4)} within policy limits`);
  } else {
    reasons.unshift(`Rejected: accumulated risk score ${riskScore.toFixed(2)} too high`);
  }

  return {
    shouldBuy,
    confidence,
    reasons,
    riskScore,
  };
}

/**
 * Convert spending model input to numeric array for ONNX inference
 * This allows the same logic to be run as an ONNX model for real zkML proofs
 */
export function spendingInputToNumeric(input: SpendingModelInput): number[] {
  return [
    input.priceUsdc,
    input.budgetUsdc,
    input.spentTodayUsdc,
    input.dailyLimitUsdc,
    input.serviceSuccessRate,
    input.serviceTotalCalls / 100, // Normalize
    input.purchasesInCategory / 10, // Normalize
    Math.min(input.timeSinceLastPurchase / 3600, 1), // Normalize to hours, cap at 1
  ];
}

/**
 * Create default input for demo purposes
 */
export function createDefaultInput(): SpendingModelInput {
  return {
    serviceUrl: 'https://api.weather.example.com',
    serviceName: 'Weather API',
    serviceCategory: 'api',
    priceUsdc: 0.002,
    budgetUsdc: 1.50,
    spentTodayUsdc: 0.15,
    dailyLimitUsdc: 1.0,
    serviceSuccessRate: 0.95,
    serviceTotalCalls: 12,
    purchasesInCategory: 3,
    timeSinceLastPurchase: 300,
  };
}
