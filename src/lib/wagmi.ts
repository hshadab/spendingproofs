import { http, createConfig } from 'wagmi';
import { defineChain } from 'viem';
import { injected, walletConnect } from 'wagmi/connectors';

// Define Arc Testnet chain
export const arcTestnet = defineChain({
  id: 5042002,
  name: 'Arc Testnet',
  nativeCurrency: {
    name: 'Arc',
    symbol: 'ARC',
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
  contracts: {
    // Add any predeployed contracts here
  },
  testnet: true,
});

// Contract addresses
export const CONTRACTS = {
  proofAttestation: (process.env.NEXT_PUBLIC_PROOF_ATTESTATION || '0xBE9a5DF7C551324CB872584C6E5bF56799787952') as `0x${string}`,
  arcAgent: (process.env.NEXT_PUBLIC_ARC_AGENT || '0x982Cd9663EBce3eB8Ab7eF511a6249621C79E384') as `0x${string}`,
  usdc: (process.env.NEXT_PUBLIC_USDC_ADDRESS || '0x0000000000000000000000000000000000000000') as `0x${string}`,
  spendingGate: process.env.NEXT_PUBLIC_SPENDING_GATE_ADDRESS as `0x${string}` | undefined,
  // Demo merchant address for receiving payments
  demoMerchant: (process.env.NEXT_PUBLIC_DEMO_MERCHANT || '0x8ba1f109551bD432803012645Ac136ddd64DBA72') as `0x${string}`,
} as const;

// Create wagmi config with SSR support
export const wagmiConfig = createConfig({
  chains: [arcTestnet],
  connectors: [
    injected(),
    ...(process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID
      ? [
          walletConnect({
            projectId: process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID,
            metadata: {
              name: 'Spending Proofs',
              description: 'zkML spending policy proofs for autonomous agents on Arc',
              url: 'https://spendingproofs.arc.network',
              icons: ['https://spendingproofs.arc.network/icon.png'],
            },
          }),
        ]
      : []),
  ],
  transports: {
    [arcTestnet.id]: http(),
  },
  ssr: true, // Enable SSR support
});

// Explorer URL helpers
export function getExplorerUrl(type: 'tx' | 'address' | 'block', value: string): string {
  const baseUrl = arcTestnet.blockExplorers.default.url;
  switch (type) {
    case 'tx':
      return `${baseUrl}/tx/${value}`;
    case 'address':
      return `${baseUrl}/address/${value}`;
    case 'block':
      return `${baseUrl}/block/${value}`;
    default:
      return baseUrl;
  }
}

// Format address for display
export function formatAddress(address: string, chars = 4): string {
  return `${address.slice(0, chars + 2)}...${address.slice(-chars)}`;
}
