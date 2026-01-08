/**
 * OpenMind Robot Spending Proofs - Library Exports
 */

export * from './types';
export * from './mockData';
export {
  generateRobotPaymentProof,
  generateMockProof,
  evaluatePaymentRequest,
  checkProverHealth,
} from './zkmlProver';
export {
  getRobotSpendingDecision,
  checkOpenMindHealth,
  type RobotDecisionRequest,
  type RobotDecisionResponse,
} from './api';
export {
  getWalletAddress,
  getWalletBalance,
  getWalletInfo,
  executeUsdcTransfer,
  simulateUsdcTransfer,
} from './wallet';
