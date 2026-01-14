/**
 * ACK-Pay Payment Service
 *
 * Issues and verifies payment receipts using W3C Verifiable Credentials
 */

import { keccak256, toBytes } from 'viem';
import type { ACKPaymentReceipt, VerifiableCredential } from './types';
import { getIsoTimestamp, getExpirationTimestamp, ACK_CONFIG } from './client';

/**
 * Create a payment receipt credential after a successful transfer
 */
export function createPaymentReceipt(params: {
  txHash: string;
  amount: string;
  recipient: string;
  proofHash: string;
  agentDid: string;
  ownerAddress: string;
}): ACKPaymentReceipt {
  const { txHash, amount, recipient, proofHash, agentDid, ownerAddress } = params;

  const receiptCredential = createReceiptCredential({
    txHash,
    amount,
    recipient,
    proofHash,
    agentDid,
    ownerAddress,
  });

  return {
    receiptCredential,
    txHash,
    amount,
    recipient,
    chainId: ACK_CONFIG.network.chainId,
    proofHash,
    issuedAt: Date.now(),
  };
}

/**
 * Create the verifiable credential for a payment receipt
 */
function createReceiptCredential(params: {
  txHash: string;
  amount: string;
  recipient: string;
  proofHash: string;
  agentDid: string;
  ownerAddress: string;
}): VerifiableCredential {
  const { txHash, amount, recipient, proofHash, agentDid, ownerAddress } = params;
  const issuanceDate = getIsoTimestamp();
  const expirationDate = getExpirationTimestamp();

  return {
    '@context': [
      'https://www.w3.org/2018/credentials/v1',
      'https://agentcommercekit.com/credentials/v1',
    ],
    type: ['VerifiableCredential', 'PaymentReceiptCredential'],
    issuer: ownerAddress,
    issuanceDate,
    expirationDate,
    credentialSubject: {
      id: agentDid,
      type: 'PaymentReceipt',
      payment: {
        txHash,
        amount,
        currency: 'USDC',
        decimals: 6,
        recipient,
        network: {
          chainId: ACK_CONFIG.network.chainId,
          name: ACK_CONFIG.network.name,
        },
      },
      verification: {
        proofHash,
        proofType: 'zkML-SNARK',
        policyVerified: true,
      },
      timestamp: Date.now(),
    },
    proof: {
      type: 'EcdsaSecp256k1Signature2019',
      created: issuanceDate,
      verificationMethod: `${agentDid}#controller`,
      proofPurpose: 'assertionMethod',
      proofValue: generateReceiptProofValue(txHash, proofHash, amount),
    },
  };
}

/**
 * Generate a deterministic proof value for the receipt
 */
function generateReceiptProofValue(
  txHash: string,
  proofHash: string,
  amount: string
): string {
  const message = `receipt:${txHash}:${proofHash}:${amount}:${ACK_CONFIG.network.chainId}`;
  const hash = keccak256(toBytes(message));
  return hash;
}

/**
 * Verify a payment receipt credential
 */
export function verifyPaymentReceipt(
  receipt: ACKPaymentReceipt,
  expectedProofHash: string
): { valid: boolean; error?: string } {
  const credential = receipt.receiptCredential;

  // Check credential type
  if (!credential.type.includes('PaymentReceiptCredential')) {
    return { valid: false, error: 'Invalid credential type' };
  }

  // Check proof hash matches
  if (receipt.proofHash !== expectedProofHash) {
    return { valid: false, error: 'Proof hash mismatch' };
  }

  // Check expiration
  if (credential.expirationDate) {
    const expiry = new Date(credential.expirationDate).getTime();
    if (Date.now() > expiry) {
      return { valid: false, error: 'Receipt has expired' };
    }
  }

  // Check proof exists
  if (!credential.proof) {
    return { valid: false, error: 'Missing proof' };
  }

  // Verify proof value format
  if (!credential.proof.proofValue.startsWith('0x')) {
    return { valid: false, error: 'Invalid proof format' };
  }

  // Verify the proof value matches expected
  const expectedProofValue = generateReceiptProofValue(
    receipt.txHash,
    receipt.proofHash,
    receipt.amount
  );
  if (credential.proof.proofValue !== expectedProofValue) {
    return { valid: false, error: 'Proof value verification failed' };
  }

  return { valid: true };
}

/**
 * Format receipt amount for display
 */
export function formatReceiptAmount(amount: string): string {
  const amountNum = parseFloat(amount);
  return `$${amountNum.toFixed(2)} USDC`;
}

/**
 * Get receipt summary for display
 */
export function getReceiptSummary(receipt: ACKPaymentReceipt): {
  txHash: string;
  amount: string;
  recipient: string;
  network: string;
  issuedAt: string;
  proofHash: string;
} {
  return {
    txHash: receipt.txHash,
    amount: formatReceiptAmount(receipt.amount),
    recipient: receipt.recipient,
    network: ACK_CONFIG.network.name,
    issuedAt: new Date(receipt.issuedAt).toLocaleString(),
    proofHash: receipt.proofHash,
  };
}

/**
 * Serialize receipt for storage/transmission
 */
export function serializeReceipt(receipt: ACKPaymentReceipt): string {
  return JSON.stringify(receipt);
}

/**
 * Deserialize receipt from storage
 */
export function deserializeReceipt(data: string): ACKPaymentReceipt | null {
  try {
    return JSON.parse(data) as ACKPaymentReceipt;
  } catch {
    return null;
  }
}
