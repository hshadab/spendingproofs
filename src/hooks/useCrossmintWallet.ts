'use client';

import { useState, useCallback, useEffect } from 'react';
import type { TransactionResult } from '@/lib/types';

interface WalletState {
  address: string | null;
  chain: string | null;
  balanceUsdc: string;
  loading: boolean;
  error: string | null;
}

interface TransferState {
  status: 'idle' | 'pending' | 'success' | 'error';
  transferId: string | null;
  txHash: string | null;
  error: string | null;
}

const INITIAL_WALLET_STATE: WalletState = {
  address: null,
  chain: null,
  balanceUsdc: '0',
  loading: false,
  error: null,
};

const INITIAL_TRANSFER_STATE: TransferState = {
  status: 'idle',
  transferId: null,
  txHash: null,
  error: null,
};

/**
 * Hook for interacting with Crossmint wallet via API routes
 */
export function useCrossmintWallet() {
  const [wallet, setWallet] = useState<WalletState>(INITIAL_WALLET_STATE);
  const [transfer, setTransfer] = useState<TransferState>(INITIAL_TRANSFER_STATE);

  /**
   * Fetch wallet info and balance
   */
  const fetchWallet = useCallback(async () => {
    setWallet(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await fetch('/api/crossmint/wallet');
      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch wallet');
      }

      setWallet({
        address: data.wallet.address,
        chain: data.wallet.chain,
        balanceUsdc: data.balance.usdc,
        loading: false,
        error: null,
      });

      return data.wallet;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setWallet(prev => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));
      return null;
    }
  }, []);

  /**
   * Execute a USDC transfer
   */
  const executeTransfer = useCallback(async (
    to: string,
    amount: number,
    proofHash?: string
  ): Promise<TransactionResult> => {
    setTransfer({
      status: 'pending',
      transferId: null,
      txHash: null,
      error: null,
    });

    try {
      const response = await fetch('/api/crossmint/transfer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          to,
          amount,
          proofHash,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Transfer failed');
      }

      setTransfer({
        status: 'success',
        transferId: data.transfer.id,
        txHash: data.transfer.txHash,
        error: null,
      });

      // Refresh wallet balance after transfer
      fetchWallet();

      return {
        success: true,
        txHash: data.transfer.txHash,
        transferId: data.transfer.id,
        chain: data.transfer.chain,
        amount: data.transfer.amount,
        recipient: data.transfer.to,
        // On-chain verification fields
        verifiedOnChain: data.transfer.verifiedOnChain,
        attestationTxHash: data.transfer.attestationTxHash,
        steps: data.steps,
        proofHash: data.transfer.proofHash,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setTransfer({
        status: 'error',
        transferId: null,
        txHash: null,
        error: errorMessage,
      });

      return {
        success: false,
        error: errorMessage,
      };
    }
  }, [fetchWallet]);

  /**
   * Reset transfer state
   */
  const resetTransfer = useCallback(() => {
    setTransfer(INITIAL_TRANSFER_STATE);
  }, []);

  /**
   * Auto-fetch wallet on mount
   */
  useEffect(() => {
    fetchWallet();
  }, [fetchWallet]);

  return {
    wallet,
    transfer,
    fetchWallet,
    executeTransfer,
    resetTransfer,
  };
}
