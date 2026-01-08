/**
 * Network configurations for Morpho demo
 */

import { ARC_CHAIN } from '@/lib/config';

export interface NetworkConfig {
  chainId: number;
  name: string;
  rpcUrl: string;
  blockExplorer: string;
  nativeToken: string;
  faucetUrl?: string;
  morphoBlueAddress?: string;
  morphoSpendingGateAddress?: string;
  isTestnet: boolean;
}

export const NETWORKS: Record<string, NetworkConfig> = {
  // Arc Testnet (Circle's L1) - use centralized config
  arcTestnet: {
    chainId: ARC_CHAIN.id,
    name: ARC_CHAIN.name,
    rpcUrl: ARC_CHAIN.rpcUrl,
    blockExplorer: ARC_CHAIN.explorerUrl,
    nativeToken: 'USDC',
    faucetUrl: 'https://faucet.circle.com',
    morphoBlueAddress: '0x034459863E9d2d400E4d005015cB74c2Cd584e0E',
    morphoSpendingGateAddress: '0x93BDD317371A2ab0D517cdE5e9FfCDa51247770D',
    isTestnet: true,
  },

  // Ethereum Mainnet
  mainnet: {
    chainId: 1,
    name: 'Ethereum',
    rpcUrl: 'https://eth.llamarpc.com',
    blockExplorer: 'https://etherscan.io',
    nativeToken: 'ETH',
    morphoBlueAddress: '0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb',
    isTestnet: false,
  },

  // Base Mainnet
  base: {
    chainId: 8453,
    name: 'Base',
    rpcUrl: 'https://mainnet.base.org',
    blockExplorer: 'https://basescan.org',
    nativeToken: 'ETH',
    morphoBlueAddress: '0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb',
    isTestnet: false,
  },
};

// Default network for demo
export const DEFAULT_NETWORK = NETWORKS.arcTestnet;

// Get network by chain ID
export function getNetworkByChainId(chainId: number): NetworkConfig | undefined {
  return Object.values(NETWORKS).find((n) => n.chainId === chainId);
}

// Format address for display
export function formatAddress(address: string): string {
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

// Get explorer URL for address
export function getExplorerAddressUrl(network: NetworkConfig, address: string): string {
  return `${network.blockExplorer}/address/${address}`;
}

// Get explorer URL for transaction
export function getExplorerTxUrl(network: NetworkConfig, txHash: string): string {
  return `${network.blockExplorer}/tx/${txHash}`;
}
