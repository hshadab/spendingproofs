/**
 * React Hook for Agent Status Monitoring
 */

import { useState, useEffect, useCallback } from 'react';
import type { AgentConfig, Address, SpendingPolicy, MorphoSpendingProofsConfig } from '../types';
import { MorphoSpendingProofsClient } from '../client';

/**
 * Agent status with computed fields
 */
export interface AgentStatus extends AgentConfig {
  /** Remaining daily allowance */
  remainingDailyLimit: bigint;
  /** Percentage of daily limit used */
  dailyLimitUsedPercent: number;
  /** Time until daily reset (seconds) */
  timeUntilReset: number;
  /** Associated policy (if loaded) */
  policy?: SpendingPolicy;
}

/**
 * Hook state
 */
export interface UseAgentStatusState {
  /** Agent status */
  status: AgentStatus | null;
  /** Whether loading */
  isLoading: boolean;
  /** Error message */
  error: string | null;
  /** Last update timestamp */
  lastUpdated: number | null;
}

/**
 * Hook configuration
 */
export interface UseAgentStatusConfig extends MorphoSpendingProofsConfig {
  /** Agent address to monitor */
  agentAddress: Address;
  /** Auto-refresh interval in ms (0 to disable) */
  refreshInterval?: number;
}

/**
 * React hook for monitoring agent status
 *
 * @example
 * ```tsx
 * const { status, isLoading, error, refresh } = useAgentStatus({
 *   proverUrl: 'https://prover.example.com',
 *   chainId: 1,
 *   gateAddress: '0x...',
 *   morphoAddress: '0x...',
 *   agentAddress: '0x...',
 *   refreshInterval: 30000, // Refresh every 30s
 * });
 *
 * return (
 *   <div>
 *     <p>Daily limit used: {status?.dailyLimitUsedPercent}%</p>
 *     <p>Remaining: {formatUnits(status?.remainingDailyLimit, 6)} USDC</p>
 *   </div>
 * );
 * ```
 */
export function useAgentStatus(config: UseAgentStatusConfig): UseAgentStatusState & {
  refresh: () => Promise<void>;
} {
  const [state, setState] = useState<UseAgentStatusState>({
    status: null,
    isLoading: true,
    error: null,
    lastUpdated: null,
  });

  const fetchStatus = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const client = new MorphoSpendingProofsClient(config);
      const agentConfig = await client.getAgentConfig(config.agentAddress);

      if (!agentConfig) {
        setState({
          status: null,
          isLoading: false,
          error: 'Agent not found',
          lastUpdated: Date.now(),
        });
        return;
      }

      const remainingLimit = await client.getRemainingDailyLimit(config.agentAddress);

      // Compute derived fields
      const now = Math.floor(Date.now() / 1000);
      const dayInSeconds = 86400;
      const timeUntilReset = Math.max(0, agentConfig.lastResetTimestamp + dayInSeconds - now);

      // Mock daily limit for computation (would come from policy in production)
      const dailyLimit = agentConfig.dailySpent + remainingLimit;
      const dailyLimitUsedPercent =
        dailyLimit > BigInt(0)
          ? Number((agentConfig.dailySpent * BigInt(100)) / dailyLimit)
          : 0;

      const status: AgentStatus = {
        ...agentConfig,
        remainingDailyLimit: remainingLimit,
        dailyLimitUsedPercent,
        timeUntilReset,
      };

      setState({
        status,
        isLoading: false,
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (err) {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch status',
        lastUpdated: Date.now(),
      }));
    }
  }, [config]);

  // Initial fetch
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Auto-refresh
  useEffect(() => {
    if (!config.refreshInterval || config.refreshInterval <= 0) return;

    const interval = setInterval(fetchStatus, config.refreshInterval);
    return () => clearInterval(interval);
  }, [config.refreshInterval, fetchStatus]);

  // Update time until reset every second
  useEffect(() => {
    if (!state.status) return;

    const interval = setInterval(() => {
      setState((prev) => {
        if (!prev.status) return prev;
        const newTimeUntilReset = Math.max(0, prev.status.timeUntilReset - 1);
        return {
          ...prev,
          status: {
            ...prev.status,
            timeUntilReset: newTimeUntilReset,
          },
        };
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [state.status?.lastResetTimestamp]);

  return {
    ...state,
    refresh: fetchStatus,
  };
}

/**
 * Format time until reset for display
 */
export function formatTimeUntilReset(seconds: number): string {
  if (seconds <= 0) return 'Resetting...';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}
