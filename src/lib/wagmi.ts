/**
 * Wagmi Configuration for Arc Testnet
 */

import { http, createConfig } from 'wagmi';
import { type Chain } from 'viem';
import { getDefaultConfig } from '@rainbow-me/rainbowkit';

/**
 * Arc Testnet Chain Definition
 */
export const arcTestnet: Chain = {
  id: 5042002,
  name: 'Arc Testnet',
  nativeCurrency: {
    name: 'USDC',
    symbol: 'USDC',
    decimals: 18,
  },
  rpcUrls: {
    default: {
      http: [process.env.NEXT_PUBLIC_ARC_RPC || 'https://rpc.testnet.arc.network'],
    },
  },
  blockExplorers: {
    default: {
      name: 'ArcScan',
      url: 'https://testnet.arcscan.app',
    },
  },
  testnet: true,
};

/**
 * Wagmi + RainbowKit Configuration
 */
export const config = getDefaultConfig({
  appName: 'Arc Policy Proofs',
  projectId: 'arc-policy-proofs-demo', // WalletConnect project ID (placeholder)
  chains: [arcTestnet],
  transports: {
    [arcTestnet.id]: http(),
  },
  ssr: true,
});

const EXPLORER_URL = 'https://testnet.arcscan.app';

/**
 * Get explorer URL for transaction
 */
export function getExplorerTxUrl(txHash: string): string {
  return `${EXPLORER_URL}/tx/${txHash}`;
}

/**
 * Get explorer URL for address
 */
export function getExplorerAddressUrl(address: string): string {
  return `${EXPLORER_URL}/address/${address}`;
}
