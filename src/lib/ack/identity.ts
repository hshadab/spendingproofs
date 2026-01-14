/**
 * ACK-ID Identity Service
 *
 * Creates and manages verifiable agent identities using W3C DIDs
 */

import { keccak256, toBytes } from 'viem';
import type { ACKAgentIdentity, VerifiableCredential, ServiceEndpoint } from './types';
import { generateAgentDid, getIsoTimestamp, getExpirationTimestamp, ACK_CONFIG } from './client';

/**
 * Create a new agent identity from a wallet address
 */
export function createAgentIdentity(
  walletAddress: `0x${string}`,
  agentName = 'Spending Agent'
): ACKAgentIdentity {
  const did = generateAgentDid(walletAddress);

  // Create controller credential
  const controllerCredential = createControllerCredential(did, walletAddress, agentName);

  // Define service endpoints
  const serviceEndpoints: ServiceEndpoint[] = [
    {
      id: `${did}#spending-proof`,
      type: 'SpendingProofService',
      serviceEndpoint: `${typeof window !== 'undefined' ? window.location.origin : ''}/api/prove`,
    },
    {
      id: `${did}#payments`,
      type: 'PaymentService',
      serviceEndpoint: `${typeof window !== 'undefined' ? window.location.origin : ''}/api/ack/receipt`,
    },
  ];

  return {
    did,
    controllerCredential,
    ownerAddress: walletAddress,
    name: agentName,
    serviceEndpoints,
    createdAt: Date.now(),
  };
}

/**
 * Create a controller credential proving agent ownership
 */
function createControllerCredential(
  agentDid: string,
  ownerAddress: string,
  agentName: string
): VerifiableCredential {
  const issuanceDate = getIsoTimestamp();
  const expirationDate = getExpirationTimestamp();

  return {
    '@context': [
      'https://www.w3.org/2018/credentials/v1',
      'https://agentcommercekit.com/credentials/v1',
    ],
    type: ['VerifiableCredential', 'ControllerCredential'],
    issuer: ownerAddress, // Self-issued by the owner
    issuanceDate,
    expirationDate,
    credentialSubject: {
      id: agentDid,
      type: 'AIAgent',
      name: agentName,
      controller: ownerAddress,
      network: {
        chainId: ACK_CONFIG.network.chainId,
        name: ACK_CONFIG.network.name,
      },
      capabilities: ['spending-verification', 'payment-execution'],
    },
    proof: {
      type: 'EcdsaSecp256k1Signature2019',
      created: issuanceDate,
      verificationMethod: `${agentDid}#controller`,
      proofPurpose: 'assertionMethod',
      // For demo, we use a placeholder proof value
      // In production, this would be a real cryptographic signature
      proofValue: generateProofValue(agentDid, ownerAddress),
    },
  };
}

/**
 * Generate a deterministic proof value for demo purposes
 */
function generateProofValue(agentDid: string, ownerAddress: string): string {
  const message = `${agentDid}:${ownerAddress}:${ACK_CONFIG.network.chainId}`;
  const hash = keccak256(toBytes(message));
  return hash;
}

/**
 * Verify a controller credential
 */
export function verifyControllerCredential(
  credential: VerifiableCredential,
  expectedOwner: string
): { valid: boolean; error?: string } {
  // Check credential type
  if (!credential.type.includes('ControllerCredential')) {
    return { valid: false, error: 'Invalid credential type' };
  }

  // Check issuer matches expected owner
  if (credential.issuer.toLowerCase() !== expectedOwner.toLowerCase()) {
    return { valid: false, error: 'Issuer does not match expected owner' };
  }

  // Check expiration
  if (credential.expirationDate) {
    const expiry = new Date(credential.expirationDate).getTime();
    if (Date.now() > expiry) {
      return { valid: false, error: 'Credential has expired' };
    }
  }

  // Check proof exists
  if (!credential.proof) {
    return { valid: false, error: 'Missing proof' };
  }

  // In production, verify the cryptographic proof signature
  // For demo, we just check the proof value format
  if (!credential.proof.proofValue.startsWith('0x')) {
    return { valid: false, error: 'Invalid proof format' };
  }

  return { valid: true };
}

/**
 * Extract agent DID from identity
 */
export function getAgentDid(identity: ACKAgentIdentity): string {
  return identity.did;
}

/**
 * Check if identity is valid and not expired
 */
export function isIdentityValid(identity: ACKAgentIdentity): boolean {
  const verification = verifyControllerCredential(
    identity.controllerCredential,
    identity.ownerAddress
  );
  return verification.valid;
}

/**
 * Serialize identity for storage
 */
export function serializeIdentity(identity: ACKAgentIdentity): string {
  return JSON.stringify(identity);
}

/**
 * Deserialize identity from storage
 */
export function deserializeIdentity(data: string): ACKAgentIdentity | null {
  try {
    return JSON.parse(data) as ACKAgentIdentity;
  } catch {
    return null;
  }
}
