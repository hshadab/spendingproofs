/**
 * OpenMind API Client
 *
 * Real integration with OpenMind's LLM API for robot decision making
 */

import { createLogger } from '@/lib/metrics';

const logger = createLogger('lib:openmind:api');

const OPENMIND_API_URL = 'https://api.openmind.org';

export interface RobotDecisionRequest {
  robotId: string;
  robotName: string;
  currentTask: string;
  batteryLevel: number;
  walletBalance: number;
  spentToday: number;
  dailyLimit: number;
  serviceRequest: {
    name: string;
    category: string;
    price: number;
    reliability: number;
    description: string;
  };
  policy: {
    maxSingleTx: number;
    allowedCategories: string[];
    minReliability: number;
  };
}

export interface RobotDecisionResponse {
  decision: 'approve' | 'reject';
  confidence: number;
  reasoning: string;
  riskFactors: string[];
  recommendation: string;
}

/**
 * Call OpenMind LLM to make a robot spending decision
 */
export async function getRobotSpendingDecision(
  request: RobotDecisionRequest
): Promise<RobotDecisionResponse> {
  const apiKey = process.env.OPENMIND_API_KEY;

  if (!apiKey) {
    logger.warn('OPENMIND_API_KEY not set, using mock response', { action: 'get_decision' });
    return getMockDecision(request);
  }

  const systemPrompt = `You are an AI spending policy evaluator for autonomous robots.
Your job is to evaluate payment requests against the robot's configured spending policy.

Respond ONLY with valid JSON in this exact format:
{
  "decision": "approve" or "reject",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "riskFactors": ["factor1", "factor2"],
  "recommendation": "what the robot should do"
}`;

  const userPrompt = `Robot "${request.robotName}" (ID: ${request.robotId}) is requesting to make a payment.

Current State:
- Task: ${request.currentTask}
- Battery: ${request.batteryLevel}%
- Wallet Balance: $${request.walletBalance.toFixed(2)} USDC
- Spent Today: $${request.spentToday.toFixed(2)}
- Daily Limit: $${request.dailyLimit.toFixed(2)}

Payment Request:
- Service: ${request.serviceRequest.name}
- Category: ${request.serviceRequest.category}
- Price: $${request.serviceRequest.price.toFixed(4)} USDC
- Reliability: ${(request.serviceRequest.reliability * 100).toFixed(0)}%
- Description: ${request.serviceRequest.description}

Policy Constraints:
- Max Single Transaction: $${request.policy.maxSingleTx.toFixed(2)}
- Allowed Categories: ${request.policy.allowedCategories.join(', ')}
- Min Service Reliability: ${(request.policy.minReliability * 100).toFixed(0)}%

Should this payment be approved? Evaluate against all policy constraints.`;

  try {
    const response = await fetch(`${OPENMIND_API_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.1,
        max_tokens: 500,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      logger.error('OpenMind API error', {
        action: 'get_decision',
        status: response.status,
        error: errorText
      });
      throw new Error(`OpenMind API error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      throw new Error('No response content from OpenMind');
    }

    // Parse JSON response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Could not parse JSON from response');
    }

    const parsed = JSON.parse(jsonMatch[0]) as RobotDecisionResponse;

    logger.info('OpenMind decision received', {
      action: 'get_decision',
      decision: parsed.decision,
      confidence: parsed.confidence,
    });

    return parsed;
  } catch (error) {
    logger.error('Failed to get OpenMind decision, using mock', { action: 'get_decision', error });
    return getMockDecision(request);
  }
}

/**
 * Mock decision for when API is unavailable
 */
function getMockDecision(request: RobotDecisionRequest): RobotDecisionResponse {
  const { serviceRequest, policy, spentToday, dailyLimit } = request;

  const remainingBudget = dailyLimit - spentToday;
  const withinBudget = serviceRequest.price <= remainingBudget;
  const withinTxLimit = serviceRequest.price <= policy.maxSingleTx;
  const categoryAllowed = policy.allowedCategories.includes(serviceRequest.category);
  const reliabilityMet = serviceRequest.reliability >= policy.minReliability;

  const allPassed = withinBudget && withinTxLimit && categoryAllowed && reliabilityMet;

  const riskFactors: string[] = [];
  if (!withinBudget) riskFactors.push(`Exceeds remaining daily budget ($${remainingBudget.toFixed(2)})`);
  if (!withinTxLimit) riskFactors.push(`Exceeds max transaction limit ($${policy.maxSingleTx})`);
  if (!categoryAllowed) riskFactors.push(`Category "${serviceRequest.category}" not allowed`);
  if (!reliabilityMet) riskFactors.push(`Service reliability below minimum`);

  return {
    decision: allPassed ? 'approve' : 'reject',
    confidence: allPassed ? 0.95 : 0.92,
    reasoning: allPassed
      ? `Payment of $${serviceRequest.price.toFixed(4)} for ${serviceRequest.name} meets all policy requirements.`
      : `Payment rejected due to policy violations: ${riskFactors.join(', ')}`,
    riskFactors,
    recommendation: allPassed
      ? 'Proceed with payment execution'
      : 'Do not execute payment - policy violation detected',
  };
}

/**
 * Check OpenMind API health
 */
export async function checkOpenMindHealth(): Promise<{ healthy: boolean; hasApiKey: boolean }> {
  const hasApiKey = !!process.env.OPENMIND_API_KEY;

  if (!hasApiKey) {
    return { healthy: false, hasApiKey: false };
  }

  try {
    const response = await fetch(`${OPENMIND_API_URL}/health`, {
      method: 'GET',
      headers: {
        'x-api-key': process.env.OPENMIND_API_KEY!,
      },
      signal: AbortSignal.timeout(5000),
    });
    return { healthy: response.ok, hasApiKey: true };
  } catch {
    return { healthy: false, hasApiKey: true };
  }
}
