'use client';

import { useCallback, useState } from 'react';
import type { SkyfirePayToken } from '@/lib/skyfire/types';

type Address = `0x${string}`;

export interface SkyfireTransferStep {
  step: string;
  status: 'success' | 'failed' | 'skipped' | 'pending';
  txHash?: string;
  hash?: string;
  data?: Record<string, unknown>;
}

export interface SkyfireTransferResult {
  success: boolean;
  transfer?: {
    status: string;
    txHash: string;
    explorerUrl: string;
    to: string;
    amount: number;
    skyfire: {
      agentId: string;
      agentName: string;
      kyaVerified: boolean;
    };
    zkml: {
      proofHash: string;
      verificationHash: string;
      decision: boolean;
      confidence: number;
    };
    attestation: {
      txHash?: string;
      explorerUrl?: string;
    };
    payment: {
      skyfireTransactionId?: string;
      gatedTransferTxHash: string;
    };
  };
  steps?: SkyfireTransferStep[];
  summary?: {
    identity: string;
    compliance: string;
    attestation: string;
    payment: string;
    result: string;
  };
  error?: string;
}

export interface PayTokenResult {
  success: boolean;
  payToken?: SkyfirePayToken;
  binding?: {
    identity: { type: string; agentId: string; verified: boolean };
    compliance: { type: string; proofHash?: string; verificationHash?: string; verified: boolean };
  };
  error?: string;
}

export interface UseSkyfirePaymentReturn {
  /** Generate a PAY token with proof binding */
  generatePayToken: (
    agentId: string,
    amount: number,
    recipient: Address,
    proofHash?: string,
    verificationHash?: string,
    kyaToken?: string
  ) => Promise<PayTokenResult>;

  /** Execute verified transfer (full flow) */
  executeVerifiedTransfer: (
    recipient: Address,
    amount: number,
    proofHash: string,
    verification: {
      decision: boolean;
      confidence: number;
    },
    agentId?: string,
    agentName?: string
  ) => Promise<SkyfireTransferResult>;

  /** Current PAY token */
  payToken: SkyfirePayToken | null;

  /** Transfer result */
  transferResult: SkyfireTransferResult | null;

  /** Loading states */
  isGeneratingPayToken: boolean;
  isTransferring: boolean;

  /** Errors */
  payTokenError: string | null;
  transferError: string | null;

  /** Reset state */
  reset: () => void;
}

/**
 * Hook for Skyfire payment operations
 *
 * Handles:
 * 1. PAY token generation with zkML proof binding
 * 2. Full verified transfer flow (KYA + zkML + attestation + payment)
 */
export function useSkyfirePayment(): UseSkyfirePaymentReturn {
  const [payToken, setPayToken] = useState<SkyfirePayToken | null>(null);
  const [transferResult, setTransferResult] = useState<SkyfireTransferResult | null>(null);
  const [isGeneratingPayToken, setIsGeneratingPayToken] = useState(false);
  const [isTransferring, setIsTransferring] = useState(false);
  const [payTokenError, setPayTokenError] = useState<string | null>(null);
  const [transferError, setTransferError] = useState<string | null>(null);

  // Generate PAY token with proof binding
  const generatePayToken = useCallback(
    async (
      agentId: string,
      amount: number,
      recipient: Address,
      proofHash?: string,
      verificationHash?: string,
      kyaToken?: string
    ): Promise<PayTokenResult> => {
      setIsGeneratingPayToken(true);
      setPayTokenError(null);

      try {
        const response = await fetch('/api/skyfire/pay-token', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            agentId,
            amount,
            recipient,
            proofHash,
            verificationHash,
            kyaToken,
          }),
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          const errorMsg = data.error || 'Failed to generate PAY token';
          setPayTokenError(errorMsg);
          return { success: false, error: errorMsg };
        }

        setPayToken(data.payToken);
        return {
          success: true,
          payToken: data.payToken,
          binding: data.binding,
        };
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setPayTokenError(errorMsg);
        return { success: false, error: errorMsg };
      } finally {
        setIsGeneratingPayToken(false);
      }
    },
    []
  );

  // Execute full verified transfer
  const executeVerifiedTransfer = useCallback(
    async (
      recipient: Address,
      amount: number,
      proofHash: string,
      verification: { decision: boolean; confidence: number },
      agentId?: string,
      agentName?: string
    ): Promise<SkyfireTransferResult> => {
      setIsTransferring(true);
      setTransferError(null);
      setTransferResult(null);

      try {
        const response = await fetch('/api/skyfire/transfer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            to: recipient,
            amount,
            proofHash,
            agentId,
            agentName,
            decision: verification.decision,
            confidence: verification.confidence,
            verifiedAt: Math.floor(Date.now() / 1000),
          }),
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          const errorMsg = data.error || 'Transfer failed';
          setTransferError(errorMsg);
          setTransferResult({ success: false, error: errorMsg, steps: data.steps });
          return { success: false, error: errorMsg, steps: data.steps };
        }

        const result: SkyfireTransferResult = {
          success: true,
          transfer: data.transfer,
          steps: data.steps,
          summary: data.summary,
        };

        setTransferResult(result);
        return result;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setTransferError(errorMsg);
        setTransferResult({ success: false, error: errorMsg });
        return { success: false, error: errorMsg };
      } finally {
        setIsTransferring(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setPayToken(null);
    setTransferResult(null);
    setIsGeneratingPayToken(false);
    setIsTransferring(false);
    setPayTokenError(null);
    setTransferError(null);
  }, []);

  return {
    generatePayToken,
    executeVerifiedTransfer,
    payToken,
    transferResult,
    isGeneratingPayToken,
    isTransferring,
    payTokenError,
    transferError,
    reset,
  };
}
