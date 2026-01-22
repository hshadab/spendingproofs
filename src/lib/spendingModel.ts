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
  | 'software'
  | 'infrastructure'
  | 'observability'
  | 'security'
  | 'other';

/**
 * Vendor tier classification for enterprise procurement
 */
export type VendorTier = 'preferred' | 'standard' | 'new' | 'high-risk';

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

  // === ENTERPRISE PROCUREMENT FIELDS (optional) ===

  // Vendor Risk Assessment
  vendorRiskScore?: number;           // 0-1, higher = riskier
  vendorId?: string;                  // Unique vendor identifier
  vendorTier?: VendorTier;            // Vendor classification

  // Budget Category Management
  budgetCategory?: string;            // e.g., 'observability', 'infrastructure'
  categoryBudgetUsdc?: number;        // Budget allocated for this category
  categorySpentUsdc?: number;         // Already spent in this category

  // Historical Vendor Performance
  historicalVendorScore?: number;     // 0-1, cumulative vendor quality (higher = better)
  vendorOnboardingDays?: number;      // How long vendor relationship exists
  vendorComplianceStatus?: boolean;   // Whether vendor passed compliance checks

  // Approval Context
  urgencyFlag?: boolean;              // Expedited approval needed
  managerPreApproval?: boolean;       // Pre-approved by manager
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

  // Enterprise procurement policy (optional)
  maxVendorRiskScore?: number;        // Reject vendors above this risk (default: 0.8)
  minVendorHistoryScore?: number;     // Minimum acceptable vendor score (default: 0.4)
  requireCompliance?: boolean;        // Must pass compliance (default: false)
  minVendorOnboardingDays?: number;   // Min relationship duration for auto-approval
}

// Default spending policy (simple agent use case)
export const DEFAULT_SPENDING_POLICY: SpendingPolicy = {
  dailyLimitUsdc: 1.0,         // $1/day max
  maxSinglePurchaseUsdc: 0.10, // $0.10 max per purchase
  minSuccessRate: 0.5,         // 50% success rate minimum
  minBudgetBuffer: 0.01,       // Keep $0.01 minimum in treasury
};

// Enterprise procurement policy (CFO/treasury use case)
export const ENTERPRISE_PROCUREMENT_POLICY: SpendingPolicy = {
  dailyLimitUsdc: 50000,        // $50K/day (or monthly depending on context)
  maxSinglePurchaseUsdc: 10000, // $10K max per purchase
  minSuccessRate: 0.95,         // 95% uptime/success rate minimum
  minBudgetBuffer: 1000,        // Keep $1K minimum in treasury
  maxVendorRiskScore: 0.7,      // Reject vendors with >70% risk score
  minVendorHistoryScore: 0.5,   // Require 50%+ historical performance
  requireCompliance: true,      // Must pass compliance checks
  minVendorOnboardingDays: 30,  // 30-day minimum relationship for auto-approval
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

  // === ENTERPRISE FIELD VALIDATIONS (only if provided) ===

  // Vendor risk score (0-1, higher = riskier)
  if (input.vendorRiskScore !== undefined) {
    if (typeof input.vendorRiskScore !== 'number' || isNaN(input.vendorRiskScore)) {
      errors.push('vendorRiskScore must be a valid number');
    } else if (input.vendorRiskScore < 0 || input.vendorRiskScore > 1) {
      errors.push('vendorRiskScore must be between 0 and 1');
    }
  }

  // Historical vendor score (0-1, higher = better)
  if (input.historicalVendorScore !== undefined) {
    if (typeof input.historicalVendorScore !== 'number' || isNaN(input.historicalVendorScore)) {
      errors.push('historicalVendorScore must be a valid number');
    } else if (input.historicalVendorScore < 0 || input.historicalVendorScore > 1) {
      errors.push('historicalVendorScore must be between 0 and 1');
    }
  }

  // Vendor onboarding days
  if (input.vendorOnboardingDays !== undefined) {
    if (typeof input.vendorOnboardingDays !== 'number' || input.vendorOnboardingDays < 0) {
      errors.push('vendorOnboardingDays must be non-negative');
    } else if (!Number.isInteger(input.vendorOnboardingDays)) {
      errors.push('vendorOnboardingDays must be an integer');
    }
  }

  // Category budget validation
  if (input.categoryBudgetUsdc !== undefined) {
    if (typeof input.categoryBudgetUsdc !== 'number' || isNaN(input.categoryBudgetUsdc)) {
      errors.push('categoryBudgetUsdc must be a valid number');
    } else if (input.categoryBudgetUsdc < 0) {
      errors.push('categoryBudgetUsdc must be non-negative');
    }
  }

  // Category spent validation
  if (input.categorySpentUsdc !== undefined) {
    if (typeof input.categorySpentUsdc !== 'number' || isNaN(input.categorySpentUsdc)) {
      errors.push('categorySpentUsdc must be a valid number');
    } else if (input.categorySpentUsdc < 0) {
      errors.push('categorySpentUsdc must be non-negative');
    }
  }

  // Cross-field validation: category spent cannot exceed category budget
  if (
    input.categoryBudgetUsdc !== undefined &&
    input.categorySpentUsdc !== undefined &&
    input.categorySpentUsdc > input.categoryBudgetUsdc
  ) {
    errors.push('categorySpentUsdc cannot exceed categoryBudgetUsdc');
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

  // === ENTERPRISE HARD BLOCKS (only evaluated if enterprise fields provided) ===

  // 5. Vendor risk score exceeds threshold
  const maxVendorRisk = policy.maxVendorRiskScore ?? 0.8;
  if (input.vendorRiskScore !== undefined && input.vendorRiskScore > maxVendorRisk) {
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: [`Vendor risk ${(input.vendorRiskScore * 100).toFixed(0)}% exceeds threshold ${(maxVendorRisk * 100).toFixed(0)}%`],
      riskScore: 1.0,
    };
  }

  // 6. Vendor failed compliance (if compliance required)
  if (policy.requireCompliance && input.vendorComplianceStatus === false) {
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: ['Vendor failed compliance verification'],
      riskScore: 1.0,
    };
  }

  // 7. Category budget exceeded
  if (
    input.categoryBudgetUsdc !== undefined &&
    input.categorySpentUsdc !== undefined &&
    input.priceUsdc + input.categorySpentUsdc > input.categoryBudgetUsdc
  ) {
    const projectedCategorySpend = input.categorySpentUsdc + input.priceUsdc;
    return {
      shouldBuy: false,
      confidence: 1.0,
      reasons: [`Would exceed category budget: $${projectedCategorySpend.toFixed(2)} > $${input.categoryBudgetUsdc.toFixed(2)}`],
      riskScore: 1.0,
    };
  }

  // 8. Vendor history score below minimum
  const minVendorHistory = policy.minVendorHistoryScore ?? 0.4;
  if (
    input.historicalVendorScore !== undefined &&
    input.historicalVendorScore < minVendorHistory &&
    input.vendorOnboardingDays !== undefined &&
    input.vendorOnboardingDays >= 90 // Only enforce if vendor is established (90+ days)
  ) {
    return {
      shouldBuy: false,
      confidence: 0.95,
      reasons: [`Vendor history ${(input.historicalVendorScore * 100).toFixed(0)}% below minimum ${(minVendorHistory * 100).toFixed(0)}%`],
      riskScore: 0.95,
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

  // === ENTERPRISE SOFT FACTORS ===

  // Factor 6: Vendor risk score (if provided)
  if (input.vendorRiskScore !== undefined) {
    // Add proportional risk based on vendor risk score
    riskScore += input.vendorRiskScore * 0.3;
    if (input.vendorRiskScore <= 0.3) {
      reasons.push(`Low vendor risk: ${(input.vendorRiskScore * 100).toFixed(0)}%`);
    } else if (input.vendorRiskScore >= 0.5) {
      reasons.push(`Elevated vendor risk: ${(input.vendorRiskScore * 100).toFixed(0)}%`);
    }
  }

  // Factor 7: Historical vendor performance (if provided)
  if (input.historicalVendorScore !== undefined) {
    // Good history reduces risk, poor history increases it
    if (input.historicalVendorScore >= 0.9) {
      riskScore -= 0.15;
      reasons.push(`Excellent vendor history: ${(input.historicalVendorScore * 100).toFixed(0)}%`);
    } else if (input.historicalVendorScore >= 0.7) {
      riskScore -= 0.05;
      reasons.push(`Good vendor history: ${(input.historicalVendorScore * 100).toFixed(0)}%`);
    } else if (input.historicalVendorScore < 0.5) {
      riskScore += 0.15;
      reasons.push(`Poor vendor history: ${(input.historicalVendorScore * 100).toFixed(0)}%`);
    }
  }

  // Factor 8: Vendor relationship duration
  if (input.vendorOnboardingDays !== undefined) {
    if (input.vendorOnboardingDays < 30) {
      riskScore += 0.1;
      reasons.push(`New vendor: ${input.vendorOnboardingDays} days`);
    } else if (input.vendorOnboardingDays >= 365) {
      riskScore -= 0.1;
      reasons.push(`Established vendor: ${Math.floor(input.vendorOnboardingDays / 365)}+ years`);
    }
  }

  // Factor 9: Compliance status boost
  if (input.vendorComplianceStatus === true) {
    riskScore -= 0.05;
    reasons.push('Compliance verified');
  }

  // Factor 10: Category budget utilization
  if (input.categoryBudgetUsdc !== undefined && input.categorySpentUsdc !== undefined) {
    const categoryUtilization = input.categorySpentUsdc / input.categoryBudgetUsdc;
    const remainingCategory = input.categoryBudgetUsdc - input.categorySpentUsdc;
    if (categoryUtilization > 0.8) {
      riskScore += 0.1;
      reasons.push(`Category budget ${(categoryUtilization * 100).toFixed(0)}% utilized`);
    } else {
      reasons.push(`Category budget: $${remainingCategory.toLocaleString()} remaining`);
    }
  }

  // Factor 11: Manager pre-approval boost
  if (input.managerPreApproval === true) {
    riskScore -= 0.2;
    reasons.push('Manager pre-approval on file');
  }

  // Factor 12: Urgency flag (slight risk increase for rushed decisions)
  if (input.urgencyFlag === true) {
    riskScore += 0.05;
    reasons.push('Expedited processing requested');
  }

  // Factor 13: Preferred vendor tier
  if (input.vendorTier === 'preferred') {
    riskScore -= 0.1;
    reasons.push('Preferred vendor tier');
  } else if (input.vendorTier === 'high-risk') {
    riskScore += 0.15;
    reasons.push('High-risk vendor classification');
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
 *
 * Array format (17 elements):
 * [0-7]   Base fields (price, budget, spent, limit, success rate, calls, category purchases, time)
 * [8-16]  Enterprise fields (vendor risk, vendor score, onboarding, category budget, category spent, etc.)
 */
export function spendingInputToNumeric(input: SpendingModelInput): number[] {
  return [
    // Base fields (0-7)
    input.priceUsdc,
    input.budgetUsdc,
    input.spentTodayUsdc,
    input.dailyLimitUsdc,
    input.serviceSuccessRate,
    input.serviceTotalCalls / 100, // Normalize
    input.purchasesInCategory / 10, // Normalize
    Math.min(input.timeSinceLastPurchase / 3600, 1), // Normalize to hours, cap at 1

    // Enterprise fields (8-16)
    input.vendorRiskScore ?? 0.5,                    // Default: medium risk
    input.historicalVendorScore ?? 0.5,             // Default: neutral history
    Math.min((input.vendorOnboardingDays ?? 0) / 365, 2), // Normalize to years, cap at 2
    input.categoryBudgetUsdc ?? 0,
    input.categorySpentUsdc ?? 0,
    input.vendorComplianceStatus === true ? 1 : 0,
    input.managerPreApproval === true ? 1 : 0,
    input.urgencyFlag === true ? 1 : 0,
    input.vendorTier === 'preferred' ? 1 : input.vendorTier === 'high-risk' ? -1 : 0,
  ];
}

/**
 * Create default input for demo purposes (simple agent use case)
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

/**
 * Create enterprise procurement demo input (DataDog APM scenario)
 *
 * Scenario: Enterprise agent purchasing $4,500/month DataDog APM subscription
 * - $50K monthly autonomous spending authority
 * - Established 2-year vendor relationship
 * - Low vendor risk, excellent historical performance
 * - Compliance verified
 */
export function createEnterpriseDemoInput(): SpendingModelInput {
  return {
    // Service Info
    serviceUrl: 'https://app.datadoghq.com/billing',
    serviceName: 'DataDog APM',
    serviceCategory: 'observability',
    priceUsdc: 4500.00,              // $4,500/month subscription

    // Agent Financial State
    budgetUsdc: 50000.00,            // $50K monthly budget
    spentTodayUsdc: 12500.00,        // $12.5K spent this month
    dailyLimitUsdc: 50000.00,        // Monthly limit (context: monthly budget cycle)

    // Service Reputation
    serviceSuccessRate: 0.999,       // 99.9% SLA
    serviceTotalCalls: 24,           // 2 years of monthly payments

    // Agent Behavior
    purchasesInCategory: 4,          // 4 observability vendors
    timeSinceLastPurchase: 604800,   // 1 week since last purchase

    // === ENTERPRISE PROCUREMENT FIELDS ===

    // Vendor Risk Assessment
    vendorRiskScore: 0.15,           // DataDog is low risk (well-known, public company)
    vendorId: 'datadog-inc',
    vendorTier: 'preferred',         // Preferred vendor status

    // Budget Category Management
    budgetCategory: 'observability',
    categoryBudgetUsdc: 15000.00,    // $15K for observability category
    categorySpentUsdc: 4200.00,      // $4.2K already spent in category

    // Historical Vendor Performance
    historicalVendorScore: 0.92,     // Excellent 2-year track record
    vendorOnboardingDays: 730,       // 2-year relationship
    vendorComplianceStatus: true,    // SOC2, GDPR compliant

    // Approval Context
    urgencyFlag: false,              // Normal processing
    managerPreApproval: false,       // No pre-approval needed (within authority)
  };
}

/**
 * AWS AgentCore Enterprise Spending Policy
 * Higher limits for cloud infrastructure spending
 */
export const AGENTCORE_ENTERPRISE_POLICY: SpendingPolicy = {
  dailyLimitUsdc: 100000,         // $100K/month (or daily depending on context)
  maxSinglePurchaseUsdc: 15000,   // $15K max per purchase
  minSuccessRate: 0.99,           // 99% uptime/success rate minimum
  minBudgetBuffer: 5000,          // Keep $5K minimum in treasury
  maxVendorRiskScore: 0.6,        // Reject vendors with >60% risk score
  minVendorHistoryScore: 0.6,     // Require 60%+ historical performance
  requireCompliance: true,        // Must pass compliance checks
  minVendorOnboardingDays: 365,   // 1-year minimum relationship for auto-approval
};

/**
 * Create AWS AgentCore demo input (Cloud Infrastructure Agent scenario)
 *
 * Scenario: Cloud Infrastructure Agent auto-scaling AWS compute
 * - EC2 p4d.24xlarge GPU instances for ML training
 * - $8,500 per instance, $100K monthly budget
 * - 5-year AWS relationship
 */
export function createAgentCoreDemoInput(): SpendingModelInput {
  return {
    // Service Info
    serviceUrl: 'https://console.aws.amazon.com/ec2/v2/home',
    serviceName: 'AWS EC2 p4d.24xlarge',
    serviceCategory: 'compute',
    priceUsdc: 8500.00,              // $8,500 for GPU instance allocation

    // Agent Financial State
    budgetUsdc: 100000.00,           // $100K monthly budget
    spentTodayUsdc: 34000.00,        // $34K spent this month (4 instances)
    dailyLimitUsdc: 100000.00,       // Monthly limit (context: monthly budget cycle)

    // Service Reputation
    serviceSuccessRate: 0.9999,      // 99.99% AWS SLA
    serviceTotalCalls: 60,           // 5 years of monthly usage

    // Agent Behavior
    purchasesInCategory: 8,          // 8 compute purchases this month
    timeSinceLastPurchase: 86400,    // 1 day since last purchase

    // === ENTERPRISE PROCUREMENT FIELDS ===

    // Vendor Risk Assessment
    vendorRiskScore: 0.08,           // AWS is extremely low risk (tier 1 cloud provider)
    vendorId: 'aws-amazon',
    vendorTier: 'preferred',         // Preferred vendor status

    // Budget Category Management
    budgetCategory: 'compute',
    categoryBudgetUsdc: 75000.00,    // $75K for compute category
    categorySpentUsdc: 34000.00,     // $34K already spent in category

    // Historical Vendor Performance
    historicalVendorScore: 0.96,     // Excellent 5-year track record
    vendorOnboardingDays: 1825,      // 5-year relationship
    vendorComplianceStatus: true,    // SOC2, HIPAA, FedRAMP compliant

    // Approval Context
    urgencyFlag: false,              // Normal ML training batch
    managerPreApproval: true,        // Pre-approved for compute scaling
  };
}
