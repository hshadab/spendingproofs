/**
 * Skyfire Integration Module
 *
 * Provides KYA (Know Your Agent) identity verification and payment
 * integration for AI agent commerce. When combined with our zkML proofs,
 * enables fully trustless agent-to-agent transactions.
 *
 * Skyfire answers: WHO is the agent?
 * zkML answers: DID the agent follow its spending policy?
 * Together: Complete trustless agent commerce.
 */

export * from './types';
export * from './config';
export { skyfireClient } from './client';
export {
  createAgent,
  generateKYAToken,
  generatePayToken,
  executePayment,
  verifyKYAToken,
  getAgent,
  hasCapability,
} from './client';
