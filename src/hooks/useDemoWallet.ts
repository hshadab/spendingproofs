'use client';

import { useState, useCallback, useEffect } from 'react';

interface DemoWalletStatus {
  address: string;
  nativeBalance: string;
  nativeFormatted: string;
  usdcBalance: string;
  usdcFormatted: string;
  usdcConfigured: boolean;
  spendingGateBalance: string;
  spendingGateFormatted: string;
  spendingGateConfigured: boolean;
  spendingGateAddress?: string;
  funded: boolean;
}

interface TransactionResult {
  success: boolean;
  hash?: string;
  explorerUrl?: string;
  error?: string;
  simulated?: boolean;
}

export function useDemoWallet() {
  const [status, setStatus] = useState<DemoWalletStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch wallet status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/demo/transaction');
      const data = await response.json();

      if (response.ok) {
        setStatus(data);
        setError(null);
      } else {
        setError(data.error || 'Failed to fetch wallet status');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    }
  }, []);

  // Load status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Submit attestation
  const submitAttestation = useCallback(async (params: {
    validatorAddress: string;
    agentId: number;
    requestUri: string;
    proofHash: string;
  }): Promise<TransactionResult> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/demo/transaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'attestation', params }),
      });

      const data = await response.json();
      setIsLoading(false);

      if (data.success) {
        fetchStatus(); // Refresh balance
        return data;
      } else {
        setError(data.error);
        return { success: false, error: data.error };
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Transaction failed';
      setError(errorMsg);
      setIsLoading(false);
      return { success: false, error: errorMsg };
    }
  }, [fetchStatus]);

  // Execute payment
  const executePayment = useCallback(async (params: {
    to: string;
    amount: number;
  }): Promise<TransactionResult> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/demo/transaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'payment', params }),
      });

      const data = await response.json();
      setIsLoading(false);

      if (data.success || data.simulated) {
        fetchStatus(); // Refresh balance
        return data;
      } else {
        setError(data.error);
        return { success: false, error: data.error };
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Payment failed';
      setError(errorMsg);
      setIsLoading(false);
      return { success: false, error: errorMsg };
    }
  }, [fetchStatus]);

  // Execute gated transfer (real on-chain enforcement)
  const executeGatedTransfer = useCallback(async (params: {
    to: string;
    amount: number;
    proofHash: string;
    expiry?: number;
  }): Promise<TransactionResult & { reverted?: boolean; revertReason?: string }> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/demo/transaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'gatedTransfer', params }),
      });

      const data = await response.json();
      setIsLoading(false);

      if (data.success) {
        fetchStatus(); // Refresh balance
        return data;
      } else {
        // Return revert info for enforcement demo
        return {
          success: false,
          reverted: data.reverted,
          revertReason: data.revertReason,
          error: data.error,
        };
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Gated transfer failed';
      setError(errorMsg);
      setIsLoading(false);
      return { success: false, error: errorMsg };
    }
  }, [fetchStatus]);

  return {
    status,
    isLoading,
    error,
    fetchStatus,
    submitAttestation,
    executePayment,
    executeGatedTransfer,
    isConnected: !!status?.address,
    address: status?.address,
    balance: status ? {
      native: status.nativeFormatted,
      usdc: status.usdcFormatted,
      display: `${status.usdcFormatted} USDC`,
    } : null,
    spendingGate: status?.spendingGateConfigured ? {
      balance: status.spendingGateFormatted,
      address: status.spendingGateAddress,
      configured: true,
    } : null,
  };
}
