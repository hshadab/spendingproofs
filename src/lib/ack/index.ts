/**
 * Agent Commerce Kit Module
 *
 * Exports ACK-ID and ACK-Pay functionality for the demo
 */

// Types
export * from './types';

// Client configuration
export * from './client';

// Identity (ACK-ID)
export {
  createAgentIdentity,
  verifyControllerCredential,
  getAgentDid,
  isIdentityValid,
  serializeIdentity,
  deserializeIdentity,
} from './identity';

// Payments (ACK-Pay)
export {
  createPaymentReceipt,
  verifyPaymentReceipt,
  formatReceiptAmount,
  getReceiptSummary,
  serializeReceipt,
  deserializeReceipt,
} from './payments';
