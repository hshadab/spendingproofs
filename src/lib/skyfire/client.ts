/**
 * Skyfire API Client
 *
 * Client for Skyfire KYAPay token operations.
 * Supports both live API calls and demo mode for testing.
 *
 * Skyfire Token Types:
 * - "kya": Know Your Agent - identity verification only
 * - "pay": Payment authorization only
 * - "kya+pay": Combined identity + payment token
 *
 * Our zkML integration adds proof binding to demonstrate
 * policy compliance before payment.
 */

import { createLogger } from '../metrics';
import { SKYFIRE_CONFIG, DEMO_CONFIG, DEMO_CAPABILITIES, SKYFIRE_SELLER_SERVICE_ID, isSkyfireConfigured, getSkyfireHeaders } from './config';
import type {
  SkyfireAgent,
  SkyfireKYAToken,
  SkyfireKYACredentials,
  SkyfirePayToken,
  SkyfirePaymentRequest,
  SkyfirePaymentResult,
} from './types';

const logger = createLogger('lib:skyfire');

/**
 * Skyfire Create Token Request
 */
interface SkyfireCreateTokenRequest {
  type: 'kya' | 'pay' | 'kya+pay';
  buyerTag?: string;
  tokenAmount?: string;
  sellerServiceId?: string;
  sellerDomainOrUrl?: string;
  expiresAt?: number;
  identityPermissions?: string[];
}

/**
 * Skyfire Token Response from Create Token API
 */
interface SkyfireTokenResponse {
  token: string;
  expiresAt?: number;
  tokenId?: string;
}

/**
 * Skyfire Token Introspection Response
 */
interface SkyfireIntrospectResponse {
  active: boolean;
  token_type?: string;
  sub?: string;
  iss?: string;
  exp?: number;
  iat?: number;
  buyer_tag?: string;
  token_amount?: string;
}

/**
 * Skyfire Charge Token Request
 */
interface SkyfireChargeRequest {
  token: string;
  amount?: string;
}

/**
 * Get the Skyfire API base URL
 */
function getApiBaseUrl(): string {
  // Skyfire API uses /api/v1 prefix
  const baseUrl = SKYFIRE_CONFIG.apiUrl.replace(/\/$/, '');
  return `${baseUrl}/api/v1`;
}

/**
 * Create or retrieve a Skyfire agent identity
 * Uses the API key as the identity foundation
 */
export async function createAgent(
  name: string = DEMO_CONFIG.demoAgentName,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<SkyfireAgent> {
  const now = Date.now();

  if (demoMode) {
    logger.info('Creating demo Skyfire agent', { name });
    await simulateDelay(DEMO_CONFIG.kyaVerificationDelay);

    return {
      id: DEMO_CONFIG.demoAgentId,
      name,
      walletAddress: DEMO_CONFIG.demoWalletAddress,
      createdAt: new Date(now).toISOString(),
      kyaCredentials: {
        agentId: DEMO_CONFIG.demoAgentId,
        issuer: 'skyfire.xyz',
        issuedAt: now,
        expiresAt: now + 24 * 60 * 60 * 1000,
        capabilities: DEMO_CAPABILITIES,
        verificationStatus: 'verified',
      },
    };
  }

  // For live mode, create agent identity from API key
  logger.info('Creating Skyfire agent from API key', { name });

  const apiKey = process.env.SKYFIRE_API_KEY || '';
  const agentId = `skyfire-${apiKey.substring(0, 8)}`;

  return {
    id: agentId,
    name,
    walletAddress: DEMO_CONFIG.demoWalletAddress,
    createdAt: new Date(now).toISOString(),
    kyaCredentials: {
      agentId,
      issuer: 'api-sandbox.skyfire.xyz',
      issuedAt: now,
      expiresAt: now + 24 * 60 * 60 * 1000,
      capabilities: DEMO_CAPABILITIES,
      verificationStatus: 'verified',
    },
  };
}

/**
 * Generate a KYA (Know Your Agent) token for identity verification
 * Uses Skyfire's Create Token API with type "kya"
 */
export async function generateKYAToken(
  agentId: string,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<SkyfireKYAToken> {
  if (demoMode) {
    logger.info('Generating demo KYA token', { agentId });
    await simulateDelay(500);

    const now = Math.floor(Date.now() / 1000);
    const expiresAt = now + 3600;

    const claims = {
      sub: agentId,
      iss: 'skyfire.xyz',
      iat: now,
      exp: expiresAt,
      capabilities: DEMO_CAPABILITIES,
    };

    const header = btoa(JSON.stringify({ alg: 'ES256', typ: 'JWT' }));
    const payload = btoa(JSON.stringify(claims));
    const signature = btoa('demo-signature-' + agentId);

    return {
      token: `${header}.${payload}.${signature}`,
      agentId,
      expiresAt: expiresAt * 1000,
      claims,
    };
  }

  // Live API call - Create a KYA token
  logger.info('Creating Skyfire KYA token via API', { agentId });

  try {
    const requestBody: SkyfireCreateTokenRequest = {
      type: 'kya',
      buyerTag: agentId,
      sellerServiceId: SKYFIRE_SELLER_SERVICE_ID,
      expiresAt: Math.floor(Date.now() / 1000) + 300, // 5 minutes
    };

    const response = await fetch(`${getApiBaseUrl()}/tokens`, {
      method: 'POST',
      headers: getSkyfireHeaders(),
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      logger.warn('Skyfire KYA token creation failed, using fallback', {
        status: response.status,
        error: errorText
      });
      return generateKYAToken(agentId, true);
    }

    const data: SkyfireTokenResponse = await response.json();
    const now = Math.floor(Date.now() / 1000);

    return {
      token: data.token,
      agentId,
      expiresAt: data.expiresAt ? data.expiresAt * 1000 : (now + 3600) * 1000,
      claims: {
        sub: agentId,
        iss: 'api-sandbox.skyfire.xyz',
        iat: now,
        exp: data.expiresAt || now + 3600,
        capabilities: DEMO_CAPABILITIES,
      },
    };
  } catch (error) {
    logger.warn('Skyfire API error, falling back to demo mode', { error });
    return generateKYAToken(agentId, true);
  }
}

/**
 * Generate a PAY token for payment authorization
 * Uses Skyfire's Create Token API with type "kya+pay"
 * Includes zkML proof hash in buyer tag for verification
 */
export async function generatePayToken(
  request: SkyfirePaymentRequest,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<SkyfirePayToken> {
  if (demoMode) {
    logger.info('Generating demo PAY token', {
      agentId: request.agentId,
      amount: request.amount,
      hasProof: !!request.proofHash,
    });
    await simulateDelay(DEMO_CONFIG.payTokenGenerationDelay);

    const expiresAt = Date.now() + 5 * 60 * 1000;

    return {
      token: `pay_${request.agentId}_${Date.now()}_${Math.random().toString(36).substring(7)}`,
      agentId: request.agentId,
      amount: request.amount,
      currency: request.currency || 'USDC',
      recipient: request.recipient,
      expiresAt,
      proofHash: request.proofHash,
      verificationHash: request.verificationHash,
    };
  }

  // Live API call - Create a kya+pay token
  logger.info('Creating Skyfire PAY token via API', {
    agentId: request.agentId,
    amount: request.amount,
  });

  try {
    // Include proof hash in buyer tag for traceability
    const buyerTag = request.proofHash
      ? `${request.agentId}:proof:${request.proofHash.substring(0, 16)}`
      : request.agentId;

    const requestBody: SkyfireCreateTokenRequest = {
      type: 'kya+pay',
      buyerTag,
      tokenAmount: request.amount.toFixed(3), // Skyfire uses string amounts
      sellerServiceId: SKYFIRE_SELLER_SERVICE_ID,
      expiresAt: Math.floor(Date.now() / 1000) + 300, // 5 minutes
    };

    const response = await fetch(`${getApiBaseUrl()}/tokens`, {
      method: 'POST',
      headers: getSkyfireHeaders(),
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      logger.warn('Skyfire PAY token creation failed, using fallback', {
        status: response.status,
        error: errorText
      });
      return generatePayToken(request, true);
    }

    const data: SkyfireTokenResponse = await response.json();

    return {
      token: data.token,
      agentId: request.agentId,
      amount: request.amount,
      currency: request.currency || 'USDC',
      recipient: request.recipient,
      expiresAt: data.expiresAt ? data.expiresAt * 1000 : Date.now() + 5 * 60 * 1000,
      proofHash: request.proofHash,
      verificationHash: request.verificationHash,
    };
  } catch (error) {
    logger.warn('Skyfire API error, falling back to demo mode', { error });
    return generatePayToken(request, true);
  }
}

/**
 * Execute a payment via Skyfire (Charge Token)
 * In demo mode, simulates the payment
 */
export async function executePayment(
  payToken: string,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<SkyfirePaymentResult> {
  if (demoMode) {
    logger.info('Executing demo payment', { payToken: payToken.substring(0, 20) + '...' });
    await simulateDelay(DEMO_CONFIG.transferExecutionDelay);

    const transactionId = `skyfire_tx_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    const txHash = `0x${Array.from({ length: 64 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('')}`;

    return {
      success: true,
      transactionId,
      txHash,
      status: 'completed',
    };
  }

  // Live API call - Charge the token
  logger.info('Charging Skyfire token via API');

  try {
    const chargeRequest: SkyfireChargeRequest = {
      token: payToken,
    };

    const response = await fetch(`${getApiBaseUrl()}/tokens/charge`, {
      method: 'POST',
      headers: getSkyfireHeaders(),
      body: JSON.stringify(chargeRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      logger.warn('Skyfire charge failed', { status: response.status, error: errorText });
      return {
        success: false,
        status: 'failed',
        error: `Charge failed: ${response.status} - ${errorText}`,
      };
    }

    const data = await response.json();

    return {
      success: true,
      transactionId: data.chargeId || data.transactionId || data.id,
      txHash: data.txHash,
      status: 'completed',
    };
  } catch (error) {
    logger.warn('Skyfire charge error', { error });
    return {
      success: false,
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Verify/introspect a Skyfire token
 * Returns the decoded claims if valid
 */
export async function verifyKYAToken(
  token: string,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<{ valid: boolean; claims?: SkyfireKYAToken['claims'] }> {
  if (demoMode) {
    try {
      const [, payload] = token.split('.');
      const claims = JSON.parse(atob(payload));
      const now = Math.floor(Date.now() / 1000);

      return {
        valid: claims.exp > now,
        claims,
      };
    } catch {
      return { valid: false };
    }
  }

  // Live API call - Introspect token
  logger.info('Introspecting Skyfire token via API');

  try {
    const response = await fetch(`${getApiBaseUrl()}/tokens/introspect`, {
      method: 'POST',
      headers: getSkyfireHeaders(),
      body: JSON.stringify({ token }),
    });

    if (!response.ok) {
      return { valid: false };
    }

    const data: SkyfireIntrospectResponse = await response.json();

    return {
      valid: data.active,
      claims: data.active ? {
        sub: data.sub || data.buyer_tag || '',
        iss: data.iss || 'skyfire.xyz',
        iat: data.iat || 0,
        exp: data.exp || 0,
        capabilities: DEMO_CAPABILITIES,
      } : undefined,
    };
  } catch (error) {
    logger.warn('Skyfire introspect error', { error });
    return { valid: false };
  }
}

/**
 * Get agent by ID
 */
export async function getAgent(
  agentId: string,
  demoMode: boolean = !isSkyfireConfigured()
): Promise<SkyfireAgent | null> {
  if (demoMode && agentId === DEMO_CONFIG.demoAgentId) {
    return createAgent(DEMO_CONFIG.demoAgentName, true);
  }

  if (demoMode) {
    return null;
  }

  if (agentId.startsWith('skyfire-')) {
    return createAgent(DEMO_CONFIG.demoAgentName, false);
  }

  return null;
}

/**
 * Check if agent has required capabilities
 */
export function hasCapability(
  credentials: SkyfireKYACredentials,
  capability: string
): boolean {
  return credentials.capabilities.includes(capability);
}

/**
 * Utility to simulate network delay for demo mode
 */
function simulateDelay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Export client instance for convenience
 */
export const skyfireClient = {
  createAgent,
  generateKYAToken,
  generatePayToken,
  executePayment,
  verifyKYAToken,
  getAgent,
  hasCapability,
  isConfigured: isSkyfireConfigured,
};
