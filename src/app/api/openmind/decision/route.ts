/**
 * OpenMind Robot Decision API
 *
 * Calls OpenMind LLM to make real spending decisions
 */

import { NextRequest, NextResponse } from 'next/server';
import { getRobotSpendingDecision, type RobotDecisionRequest } from '@/lib/openmind/api';
import { getWalletBalance } from '@/lib/openmind/wallet';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('api:openmind:decision');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const {
      robotId,
      robotName,
      currentTask,
      batteryLevel,
      serviceRequest,
      policy,
      spentToday = 0,
    } = body;

    // Get real wallet balance from Arc testnet
    let walletBalance = 100; // Default
    try {
      const walletInfo = await getWalletBalance();
      walletBalance = walletInfo.balanceUsdc;
    } catch (error) {
      logger.warn('Could not fetch wallet balance, using default', { action: 'get_balance', error });
    }

    const decisionRequest: RobotDecisionRequest = {
      robotId: robotId || 'robot-001',
      robotName: robotName || 'DeliveryBot-7',
      currentTask: currentTask || 'Package delivery',
      batteryLevel: batteryLevel || 50,
      walletBalance,
      spentToday,
      dailyLimit: policy?.dailyLimit || 10,
      serviceRequest: {
        name: serviceRequest?.name || 'Unknown Service',
        category: serviceRequest?.category || 'other',
        price: serviceRequest?.price || 0,
        reliability: serviceRequest?.reliability || 0.95,
        description: serviceRequest?.description || 'Service request',
      },
      policy: {
        maxSingleTx: policy?.maxSingleTx || 2,
        allowedCategories: policy?.allowedCategories || ['charging', 'navigation', 'compute', 'data'],
        minReliability: policy?.minReliability || 0.85,
      },
    };

    logger.info('Processing robot decision request', {
      action: 'decision',
      robotId: decisionRequest.robotId,
      service: decisionRequest.serviceRequest.name,
      price: decisionRequest.serviceRequest.price,
    });

    const decision = await getRobotSpendingDecision(decisionRequest);

    return NextResponse.json({
      success: true,
      decision,
      walletBalance,
      request: {
        robotId: decisionRequest.robotId,
        service: decisionRequest.serviceRequest.name,
        price: decisionRequest.serviceRequest.price,
      },
    });
  } catch (error) {
    logger.error('Decision request failed', { action: 'decision', error });
    return NextResponse.json(
      { success: false, error: 'Failed to process decision request' },
      { status: 500 }
    );
  }
}
