/**
 * Mock data and presets for Morpho demo
 */

import { generateSecureBytes32 } from '@/lib/crypto';
import { fetchMorphoMarkets } from './morphoApi';
import { DEFAULT_NETWORK } from './networks';
import type { MarketData, SpendingPolicy, MorphoMarket } from './types';

// Static fallback markets (used when API unavailable)
export const MOCK_MARKETS: MarketData[] = [
  {
    address: '0xb323495f7e4148be5643a4ea4a8221eef163e4bccfdedc2a6f4696baacbc86cc',
    name: 'wstETH/USDC',
    supplyAPY: 5.21,
    borrowAPY: 6.83,
    totalSupply: 142_500_000,
    totalBorrow: 108_585_000,
    utilization: 76.2,
  },
  {
    address: '0x3a85e619751152991742810df6ec69ce473daef99e28a64ab2340d7b7ccfee49',
    name: 'wstETH/WETH',
    supplyAPY: 2.14,
    borrowAPY: 3.21,
    totalSupply: 89_200_000,
    totalBorrow: 59_496_400,
    utilization: 66.7,
  },
  {
    address: '0x698fe98247a40c5771537b5786b2f3f9d78eb487b4ce4d75533cd0e94d88a115',
    name: 'cbETH/USDC',
    supplyAPY: 4.87,
    borrowAPY: 6.12,
    totalSupply: 67_300_000,
    totalBorrow: 53_503_500,
    utilization: 79.5,
  },
];

// Fetch markets - tries live API first, falls back to mock
let cachedMarkets: MorphoMarket[] | null = null;
let lastFetchTime = 0;
const CACHE_TTL = 60000; // 1 minute cache

export async function getMarkets(): Promise<{ markets: MorphoMarket[]; isLive: boolean }> {
  const now = Date.now();

  // Return cached if fresh
  if (cachedMarkets && now - lastFetchTime < CACHE_TTL) {
    return { markets: cachedMarkets, isLive: cachedMarkets[0]?.isLive ?? false };
  }

  // Try to fetch live data
  const markets = await fetchMorphoMarkets(1); // Ethereum mainnet
  cachedMarkets = markets;
  lastFetchTime = now;

  return { markets, isLive: markets[0]?.isLive ?? false };
}

// Network info for display
export const NETWORK_INFO = {
  name: DEFAULT_NETWORK.name,
  chainId: DEFAULT_NETWORK.chainId,
  explorer: DEFAULT_NETWORK.blockExplorer,
  rpcUrl: DEFAULT_NETWORK.rpcUrl,
  faucet: DEFAULT_NETWORK.faucetUrl,
  nativeToken: DEFAULT_NETWORK.nativeToken,
};

export const DEFAULT_POLICY: SpendingPolicy = {
  dailyLimit: 10000,
  maxSingleTx: 5000,
  maxLTV: 70,
  minHealthFactor: 1.2,
  allowedMarkets: MOCK_MARKETS.map(m => m.address),
  requireProofForSupply: true,
  requireProofForBorrow: true,
  requireProofForWithdraw: true,
};

export const STRATEGY_PRESETS = {
  conservative: {
    name: 'Conservative Yield',
    description: 'Focus on stable yield with minimal risk',
    dailyLimit: 5000,
    maxSingleTx: 2500,
    maxLTV: 50,
    minHealthFactor: 1.5,
    color: 'green',
  },
  moderate: {
    name: 'Moderate Leverage',
    description: 'Balanced leverage for enhanced yields',
    dailyLimit: 10000,
    maxSingleTx: 5000,
    maxLTV: 70,
    minHealthFactor: 1.2,
    color: 'blue',
  },
  aggressive: {
    name: 'Aggressive Yield',
    description: 'Maximum yield with higher risk tolerance',
    dailyLimit: 25000,
    maxSingleTx: 10000,
    maxLTV: 85,
    minHealthFactor: 1.1,
    color: 'purple',
  },
};

export const AGENT_DECISIONS = [
  {
    operation: 0, // SUPPLY
    reasoning: 'High supply APY (5.21%) detected in USDC/wstETH market. Deploying capital to capture yield.',
    confidence: 0.92,
  },
  {
    operation: 1, // BORROW
    reasoning: 'Current LTV at 45%, below target of 70%. Adding leverage to increase yield.',
    confidence: 0.78,
  },
  {
    operation: 3, // REPAY
    reasoning: 'LTV drifted to 73%, above target. Deleveraging to maintain safety margin.',
    confidence: 0.95,
  },
  {
    operation: 2, // WITHDRAW
    reasoning: 'Profit target reached. Taking partial profits while maintaining core position.',
    confidence: 0.85,
  },
];

// Use secure random generation from crypto utils
export function generateMockTxHash(): `0x${string}` {
  return generateSecureBytes32();
}

export function generateMockProofHash(): `0x${string}` {
  return generateSecureBytes32();
}
