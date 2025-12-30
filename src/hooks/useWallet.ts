'use client';

import { useAccount, useBalance, useChainId, useSwitchChain, useDisconnect } from 'wagmi';
import { useConnectModal, useAccountModal, useChainModal } from '@rainbow-me/rainbowkit';
import { arcTestnet, CONTRACTS, formatAddress, getExplorerUrl } from '@/lib/wagmi';
import { useMemo } from 'react';

export interface WalletState {
  // Connection state
  address: `0x${string}` | undefined;
  isConnected: boolean;
  isConnecting: boolean;
  isDisconnected: boolean;

  // Chain state
  chainId: number | undefined;
  isCorrectChain: boolean;
  chainName: string;

  // Balance state
  nativeBalance: string | undefined;
  usdcBalance: string | undefined;
  usdcBalanceFormatted: string | undefined;

  // Display helpers
  displayAddress: string | undefined;
  explorerUrl: string | undefined;
}

export function useWallet() {
  const { address, isConnected, isConnecting, isDisconnected } = useAccount();
  const chainId = useChainId();
  const { switchChain, isPending: isSwitchingChain } = useSwitchChain();
  const { disconnect } = useDisconnect();

  // RainbowKit modals
  const { openConnectModal } = useConnectModal();
  const { openAccountModal } = useAccountModal();
  const { openChainModal } = useChainModal();

  // Native balance
  const { data: nativeBalanceData } = useBalance({
    address,
  });

  // USDC balance
  const { data: usdcBalanceData, refetch: refetchUsdcBalance } = useBalance({
    address,
    token: CONTRACTS.usdc !== '0x0000000000000000000000000000000000000000' ? CONTRACTS.usdc : undefined,
  });

  // Computed state
  const isCorrectChain = chainId === arcTestnet.id;

  const walletState: WalletState = useMemo(() => ({
    address,
    isConnected,
    isConnecting,
    isDisconnected,
    chainId,
    isCorrectChain,
    chainName: isCorrectChain ? arcTestnet.name : 'Unknown',
    nativeBalance: nativeBalanceData?.formatted,
    usdcBalance: usdcBalanceData?.value?.toString(),
    usdcBalanceFormatted: usdcBalanceData?.formatted,
    displayAddress: address ? formatAddress(address) : undefined,
    explorerUrl: address ? getExplorerUrl('address', address) : undefined,
  }), [
    address,
    isConnected,
    isConnecting,
    isDisconnected,
    chainId,
    isCorrectChain,
    nativeBalanceData?.formatted,
    usdcBalanceData?.value,
    usdcBalanceData?.formatted,
  ]);

  // Actions
  const connect = () => {
    openConnectModal?.();
  };

  const openAccount = () => {
    openAccountModal?.();
  };

  const openChain = () => {
    openChainModal?.();
  };

  const ensureCorrectChain = async (): Promise<boolean> => {
    if (isCorrectChain) return true;

    try {
      await switchChain({ chainId: arcTestnet.id });
      return true;
    } catch (error) {
      console.error('Failed to switch chain:', error);
      return false;
    }
  };

  const disconnectWallet = () => {
    disconnect();
  };

  return {
    // State
    ...walletState,

    // Actions
    connect,
    disconnect: disconnectWallet,
    openAccount,
    openChain,
    ensureCorrectChain,
    refetchUsdcBalance,

    // Loading states
    isSwitchingChain,

    // Constants
    arcTestnet,
    contracts: CONTRACTS,
  };
}

// Re-export for convenience
export { formatAddress, getExplorerUrl } from '@/lib/wagmi';
