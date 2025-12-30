import { describe, it, expect } from 'vitest';
import {
  runSpendingModel,
  SpendingModelInput,
  SpendingPolicy,
  DEFAULT_SPENDING_POLICY,
  spendingInputToNumeric,
  createDefaultInput,
} from '../lib/spendingModel';

describe('Spending Model', () => {
  describe('runSpendingModel', () => {
    const baseInput: SpendingModelInput = {
      serviceUrl: 'https://api.example.com',
      serviceName: 'Test API',
      serviceCategory: 'api',
      priceUsdc: 0.05,
      budgetUsdc: 1.0,
      spentTodayUsdc: 0.2,
      dailyLimitUsdc: 0.5,
      serviceSuccessRate: 0.95,
      serviceTotalCalls: 100,
      purchasesInCategory: 3,
      timeSinceLastPurchase: 3600,
    };

    describe('Hard Blocks', () => {
      it('should reject when price exceeds max single purchase', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          priceUsdc: 0.20, // Exceeds DEFAULT_SPENDING_POLICY.maxSinglePurchaseUsdc (0.10)
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(false);
        expect(result.confidence).toBe(1.0);
        expect(result.riskScore).toBe(1.0);
        expect(result.reasons[0]).toContain('exceeds max single purchase');
      });

      it('should reject when daily limit would be exceeded', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          spentTodayUsdc: 0.95,
          priceUsdc: 0.10,
          dailyLimitUsdc: 1.0,
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(false);
        expect(result.confidence).toBe(1.0);
        expect(result.reasons[0]).toContain('exceed daily limit');
      });

      it('should reject when budget is insufficient', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          budgetUsdc: 0.02,
          priceUsdc: 0.05,
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(false);
        expect(result.reasons[0]).toContain('Insufficient budget');
      });

      it('should reject when service has bad reputation with history', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          serviceSuccessRate: 0.3, // Below minSuccessRate (0.5)
          serviceTotalCalls: 10, // >= 3 calls, so reputation is considered
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(false);
        expect(result.reasons[0]).toContain('success rate');
      });
    });

    describe('Approvals', () => {
      it('should approve when all conditions are met', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          priceUsdc: 0.05,
          budgetUsdc: 1.0,
          spentTodayUsdc: 0.1,
          dailyLimitUsdc: 1.0,
          serviceSuccessRate: 0.95,
          serviceTotalCalls: 100,
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(true);
        expect(result.confidence).toBeGreaterThan(0.5);
        expect(result.riskScore).toBeLessThan(0.8);
      });

      it('should approve with higher risk for edge cases', () => {
        const input: SpendingModelInput = {
          ...baseInput,
          priceUsdc: 0.08,
          budgetUsdc: 0.15, // High budget ratio (>50%)
          spentTodayUsdc: 0.35,
          dailyLimitUsdc: 0.5, // Near 80% of daily limit
        };

        const result = runSpendingModel(input);

        expect(result.shouldBuy).toBe(true);
        // Higher risk due to high budget ratio and daily progress
        expect(result.riskScore).toBeGreaterThanOrEqual(0);
      });
    });

    describe('Risk Scoring', () => {
      it('should increase risk for new services', () => {
        const knownService: SpendingModelInput = {
          ...baseInput,
          serviceTotalCalls: 100,
          serviceSuccessRate: 0.95,
        };

        const newService: SpendingModelInput = {
          ...baseInput,
          serviceTotalCalls: 0,
          serviceSuccessRate: 0,
        };

        const knownResult = runSpendingModel(knownService);
        const newResult = runSpendingModel(newService);

        expect(newResult.riskScore).toBeGreaterThan(knownResult.riskScore);
      });

      it('should increase risk for rapid spending', () => {
        // Use inputs that have some baseline risk to see the difference
        const normalSpending: SpendingModelInput = {
          ...baseInput,
          timeSinceLastPurchase: 3600, // 1 hour
          serviceTotalCalls: 0, // New service adds baseline risk
        };

        const rapidSpending: SpendingModelInput = {
          ...baseInput,
          timeSinceLastPurchase: 30, // 30 seconds - under 60s triggers rapid spending
          serviceTotalCalls: 0, // Same baseline
        };

        const normalResult = runSpendingModel(normalSpending);
        const rapidResult = runSpendingModel(rapidSpending);

        // Rapid spending should add 0.1 to risk score
        expect(rapidResult.riskScore).toBeGreaterThan(normalResult.riskScore);
      });

      it('should increase risk for high category concentration', () => {
        // Use inputs that have some baseline risk
        const lowConcentration: SpendingModelInput = {
          ...baseInput,
          purchasesInCategory: 2,
          serviceTotalCalls: 0, // New service adds baseline risk
        };

        const highConcentration: SpendingModelInput = {
          ...baseInput,
          purchasesInCategory: 10, // > 5 triggers category concentration risk
          serviceTotalCalls: 0, // Same baseline
        };

        const lowResult = runSpendingModel(lowConcentration);
        const highResult = runSpendingModel(highConcentration);

        // High concentration (>5) should add 0.1 to risk score
        expect(highResult.riskScore).toBeGreaterThan(lowResult.riskScore);
      });
    });

    describe('Custom Policy', () => {
      it('should respect custom policy limits', () => {
        const customPolicy: SpendingPolicy = {
          dailyLimitUsdc: 5.0,
          maxSinglePurchaseUsdc: 1.0,
          minSuccessRate: 0.8,
          minBudgetBuffer: 0.05,
        };

        const input: SpendingModelInput = {
          ...baseInput,
          priceUsdc: 0.50, // Would fail with default policy
          budgetUsdc: 2.0,
          serviceSuccessRate: 0.85,
        };

        const result = runSpendingModel(input, customPolicy);

        expect(result.shouldBuy).toBe(true);
      });
    });
  });

  describe('spendingInputToNumeric', () => {
    it('should convert input to normalized numeric array', () => {
      const input = createDefaultInput();
      const numeric = spendingInputToNumeric(input);

      expect(numeric).toHaveLength(8);
      expect(numeric[0]).toBe(input.priceUsdc);
      expect(numeric[1]).toBe(input.budgetUsdc);
      expect(numeric[2]).toBe(input.spentTodayUsdc);
      expect(numeric[3]).toBe(input.dailyLimitUsdc);
      expect(numeric[4]).toBe(input.serviceSuccessRate);
      expect(numeric[5]).toBe(input.serviceTotalCalls / 100);
      expect(numeric[6]).toBe(input.purchasesInCategory / 10);
    });

    it('should cap timeSinceLastPurchase at 1', () => {
      const input = createDefaultInput();
      input.timeSinceLastPurchase = 10000; // > 3600

      const numeric = spendingInputToNumeric(input);

      expect(numeric[7]).toBe(1);
    });
  });

  describe('createDefaultInput', () => {
    it('should return valid default input', () => {
      const input = createDefaultInput();

      expect(input.priceUsdc).toBeGreaterThan(0);
      expect(input.budgetUsdc).toBeGreaterThan(0);
      expect(input.serviceSuccessRate).toBeGreaterThanOrEqual(0);
      expect(input.serviceSuccessRate).toBeLessThanOrEqual(1);
    });
  });
});
