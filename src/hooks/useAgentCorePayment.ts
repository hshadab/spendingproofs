'use client';

import { useState, useCallback, useEffect } from 'react';
import type { TransactionResult } from '@/lib/types';

interface WalletState {
  address: string | null;
  balance: string;
  chain: string;
  loading: boolean;
  error: string | null;
}

interface TransferState {
  status: 'idle' | 'pending' | 'success' | 'error';
  txHash: string | null;
  attestationTxHash: string | null;
  error: string | null;
}

interface TransferStep {
  step: string;
  status: 'success' | 'failed' | 'skipped' | 'pending';
  details?: string;
  txHash?: string;
  timeMs?: number;
}

/**
 * Proof data from MCP gateway response
 */
export interface McpProofResponse {
  jsonrpc: string;
  id: number;
  result?: {
    content?: Array<{
      type: string;
      text: string;
    }>;
  };
}

const INITIAL_WALLET_STATE: WalletState = {
  address: null,
  balance: '0',
  chain: 'arc-testnet',
  loading: false,
  error: null,
};

const INITIAL_TRANSFER_STATE: TransferState = {
  status: 'idle',
  txHash: null,
  attestationTxHash: null,
  error: null,
};

/**
 * Hook for managing AgentCore payment flow
 *
 * Handles:
 * - Wallet state (address, balance)
 * - Transfer execution with MCP proof verification
 * - Step tracking for UI
 */
export function useAgentCorePayment() {
  const [wallet, setWallet] = useState<WalletState>(INITIAL_WALLET_STATE);
  const [transfer, setTransfer] = useState<TransferState>(INITIAL_TRANSFER_STATE);
  const [steps, setSteps] = useState<TransferStep[]>([]);
  const [isSimulated, setIsSimulated] = useState<boolean | null>(null);

  /**
   * Fetch wallet info from API
   */
  const fetchWallet = useCallback(async () => {
    setWallet(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await fetch('/api/agentcore/transfer');
      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch wallet');
      }

      setIsSimulated(data.simulated);
      setWallet({
        address: data.wallet.address,
        balance: data.wallet.balance,
        chain: data.wallet.chain,
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
   * Execute USDC transfer with MCP proof verification
   *
   * @param to - Recipient address
   * @param amount - Amount in USDC
   * @param mcpProofResponse - Full MCP gateway response containing proof
   * @param proofHash - Direct proof hash (if not using mcpProofResponse)
   * @param skipVerification - Skip verification for demo
   */
  const executeTransfer = useCallback(async (
    to: string,
    amount: number,
    mcpProofResponse?: McpProofResponse | null,
    proofHash?: string,
    skipVerification?: boolean
  ): Promise<TransactionResult & {
    steps?: TransferStep[];
    attestationTxHash?: string;
    explorerUrl?: string;
    attestationExplorerUrl?: string;
  }> => {
    setTransfer({
      status: 'pending',
      txHash: null,
      attestationTxHash: null,
      error: null,
    });
    setSteps([]);

    try {
      const response = await fetch('/api/agentcore/transfer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          to,
          amount,
          mcpResponse: mcpProofResponse,
          proofHash,
          skipVerification,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Transfer failed');
      }

      setIsSimulated(data.simulated);
      setSteps(data.steps || []);
      setTransfer({
        status: 'success',
        txHash: data.transfer.txHash,
        attestationTxHash: data.transfer.attestationTxHash,
        error: null,
      });

      // Refresh wallet balance after transfer
      fetchWallet();

      return {
        success: true,
        txHash: data.transfer.txHash,
        chain: data.transfer.chain || 'arc-testnet',
        amount: data.transfer.amount,
        recipient: data.transfer.to,
        steps: data.steps,
        attestationTxHash: data.transfer.attestationTxHash,
        proofHash: data.transfer.proofHash,
        verifiedOnChain: !!data.transfer.attestationTxHash,
        method: data.transfer.method,
        explorerUrl: data.transfer.explorerUrl,
        attestationExplorerUrl: data.transfer.attestationExplorerUrl,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setTransfer({
        status: 'error',
        txHash: null,
        attestationTxHash: null,
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
    setSteps([]);
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
    steps,
    isSimulated,
    fetchWallet,
    executeTransfer,
    resetTransfer,
  };
}
