/**
 * Morpho Blue API integration for fetching live market data
 */

import { createLogger } from '@/lib/metrics';
import type { MorphoMarket } from './types';

const logger = createLogger('lib:morpho:api');

const MORPHO_API_URL = 'https://blue-api.morpho.org/graphql';

// GraphQL query for fetching markets
const MARKETS_QUERY = `
  query GetMarkets($chainId: Int!) {
    markets(where: { chainId_in: [$chainId] }, first: 10) {
      items {
        uniqueKey
        loanAsset {
          symbol
          address
          decimals
        }
        collateralAsset {
          symbol
          address
          decimals
        }
        state {
          supplyApy
          borrowApy
          utilization
          supplyAssets
          borrowAssets
          liquidityAssets
        }
        lltv
        oracle {
          address
        }
      }
    }
  }
`;

interface MorphoApiMarket {
  uniqueKey: string;
  loanAsset: { symbol: string; address: string; decimals: number };
  collateralAsset: { symbol: string; address: string; decimals: number };
  state: {
    supplyApy: number;
    borrowApy: number;
    utilization: number;
    supplyAssets: string;
    borrowAssets: string;
    liquidityAssets: string;
  };
  lltv: string;
  oracle: { address: string };
}

// Fetch live markets from Morpho API
export async function fetchMorphoMarkets(chainId: number = 1): Promise<MorphoMarket[]> {
  try {
    const response = await fetch(MORPHO_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: MARKETS_QUERY,
        variables: { chainId },
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();

    if (data.errors) {
      throw new Error(data.errors[0]?.message || 'GraphQL error');
    }

    const markets: MorphoApiMarket[] = data.data?.markets?.items || [];

    return markets.map((m) => ({
      address: m.uniqueKey,
      name: `${m.collateralAsset.symbol}/${m.loanAsset.symbol}`,
      loanAsset: m.loanAsset.symbol,
      collateralAsset: m.collateralAsset.symbol,
      supplyAPY: m.state.supplyApy * 100,
      borrowAPY: m.state.borrowApy * 100,
      utilization: m.state.utilization * 100,
      totalSupply: parseFloat(m.state.supplyAssets) / Math.pow(10, m.loanAsset.decimals),
      totalBorrow: parseFloat(m.state.borrowAssets) / Math.pow(10, m.loanAsset.decimals),
      lltv: parseFloat(m.lltv) / 1e18 * 100,
      isLive: true,
    }));
  } catch (error) {
    logger.warn('Failed to fetch live Morpho data, using fallback', { action: 'fetch_markets', error });
    return getFallbackMarkets();
  }
}

// Fallback mock data when API is unavailable
function getFallbackMarkets(): MorphoMarket[] {
  return [
    {
      address: '0xb323495f7e4148be5643a4ea4a8221eef163e4bccfdedc2a6f4696baacbc86cc',
      name: 'wstETH/USDC',
      loanAsset: 'USDC',
      collateralAsset: 'wstETH',
      supplyAPY: 5.21,
      borrowAPY: 6.83,
      utilization: 76.2,
      totalSupply: 142500000,
      totalBorrow: 108585000,
      lltv: 86,
      isLive: false,
    },
    {
      address: '0x3a85e619751152991742810df6ec69ce473daef99e28a64ab2340d7b7ccfee49',
      name: 'wstETH/WETH',
      loanAsset: 'WETH',
      collateralAsset: 'wstETH',
      supplyAPY: 2.14,
      borrowAPY: 3.21,
      utilization: 66.7,
      totalSupply: 89200000,
      totalBorrow: 59496400,
      lltv: 94.5,
      isLive: false,
    },
    {
      address: '0x698fe98247a40c5771537b5786b2f3f9d78eb487b4ce4d75533cd0e94d88a115',
      name: 'cbETH/USDC',
      loanAsset: 'USDC',
      collateralAsset: 'cbETH',
      supplyAPY: 4.87,
      borrowAPY: 6.12,
      utilization: 79.5,
      totalSupply: 67300000,
      totalBorrow: 53503500,
      lltv: 86,
      isLive: false,
    },
  ];
}

// Check if we can reach the Morpho API
export async function checkMorphoApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(MORPHO_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: '{ __typename }' }),
    });
    return response.ok;
  } catch {
    return false;
  }
}
