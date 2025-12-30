'use client';

import { useState, useCallback } from 'react';
import type { CircleWallet, CircleWalletSet, CircleTransaction } from '@/lib/circle';

export interface UseCircleWalletState {
  wallets: CircleWallet[];
  walletSets: CircleWalletSet[];
  isLoading: boolean;
  error: string | null;
}

export interface CreateWalletParams {
  walletSetId: string;
  blockchain: string;
  accountType?: 'SCA' | 'EOA';
}

export interface TransferParams {
  walletId: string;
  destinationAddress: string;
  amount: number;
  blockchain: string;
}

export function useCircleWallet() {
  const [wallets, setWallets] = useState<CircleWallet[]>([]);
  const [walletSets, setWalletSets] = useState<CircleWalletSet[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingTransfer, setPendingTransfer] = useState<CircleTransaction | null>(null);

  // Fetch wallet sets
  const fetchWalletSets = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/circle/wallet-sets');
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch wallet sets');
      }

      setWalletSets(data.data?.walletSets || []);
      return data.data?.walletSets || [];
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Create wallet set
  const createWalletSet = useCallback(async (name: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/circle/wallet-sets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create wallet set');
      }

      // Refresh wallet sets
      await fetchWalletSets();
      return data.data?.walletSet;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchWalletSets]);

  // Fetch wallets
  const fetchWallets = useCallback(async (walletSetId?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const url = walletSetId
        ? `/api/circle/wallets?walletSetId=${walletSetId}`
        : '/api/circle/wallets';
      const response = await fetch(url);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch wallets');
      }

      setWallets(data.data?.wallets || []);
      return data.data?.wallets || [];
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Create wallet
  const createWallet = useCallback(async (params: CreateWalletParams) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/circle/wallets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create wallet');
      }

      // Refresh wallets
      await fetchWallets(params.walletSetId);
      return data.data?.wallets?.[0];
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchWallets]);

  // Execute transfer
  const transfer = useCallback(async (params: TransferParams) => {
    setIsLoading(true);
    setError(null);
    setPendingTransfer(null);
    try {
      const response = await fetch('/api/circle/transfer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to execute transfer');
      }

      const transaction = data.data?.transaction;
      setPendingTransfer(transaction);
      return transaction;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Get transfer status
  const getTransferStatus = useCallback(async (transactionId: string) => {
    try {
      const response = await fetch(`/api/circle/transfer?id=${transactionId}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get transfer status');
      }

      const transaction = data.data?.transaction;
      setPendingTransfer(transaction);
      return transaction;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      throw err;
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Reset state
  const reset = useCallback(() => {
    setWallets([]);
    setWalletSets([]);
    setError(null);
    setPendingTransfer(null);
  }, []);

  return {
    // State
    wallets,
    walletSets,
    isLoading,
    error,
    pendingTransfer,

    // Actions
    fetchWalletSets,
    createWalletSet,
    fetchWallets,
    createWallet,
    transfer,
    getTransferStatus,
    clearError,
    reset,
  };
}
