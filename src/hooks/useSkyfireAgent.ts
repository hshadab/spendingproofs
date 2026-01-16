'use client';

import { useState, useCallback, useEffect } from 'react';
import type {
  SkyfireAgent,
  SkyfireKYAToken,
} from '@/lib/skyfire/types';

const STORAGE_KEY = 'skyfire-agent';

interface UseSkyfireAgentReturn {
  /** Current agent */
  agent: SkyfireAgent | null;
  /** Current KYA token */
  kyaToken: SkyfireKYAToken | null;
  /** Whether agent is being created */
  isCreating: boolean;
  /** Whether KYA token is being generated */
  isGeneratingToken: boolean;
  /** Error message if any */
  error: string | null;
  /** Create a new Skyfire agent */
  createAgent: (name?: string) => Promise<SkyfireAgent>;
  /** Generate a new KYA token */
  refreshKYAToken: () => Promise<SkyfireKYAToken>;
  /** Clear the current agent */
  clearAgent: () => void;
  /** Whether agent has valid KYA credentials */
  isKYAValid: boolean;
}

/**
 * Hook for managing Skyfire agent identity
 *
 * Creates and manages agent identity with KYA (Know Your Agent) credentials.
 * Skyfire provides agent identity verification - WHO is the agent.
 */
export function useSkyfireAgent(): UseSkyfireAgentReturn {
  const [agent, setAgent] = useState<SkyfireAgent | null>(null);
  const [kyaToken, setKyaToken] = useState<SkyfireKYAToken | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isGeneratingToken, setIsGeneratingToken] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load agent from storage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        // Check if KYA credentials are still valid
        if (parsed.kyaCredentials && parsed.kyaCredentials.expiresAt > Date.now()) {
          setAgent(parsed);
        } else {
          // Clear expired agent
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  // Create a new agent via API
  const createAgent = useCallback(
    async (name = 'zkML Demo Agent'): Promise<SkyfireAgent> => {
      setIsCreating(true);
      setError(null);

      try {
        const response = await fetch('/api/skyfire/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agentName: name, generateToken: true }),
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || 'Failed to create agent');
        }

        const newAgent = data.agent as SkyfireAgent;

        // Store in localStorage
        if (typeof window !== 'undefined') {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(newAgent));
        }

        setAgent(newAgent);

        // Also set KYA token if returned
        if (data.kyaToken) {
          setKyaToken(data.kyaToken);
        }

        return newAgent;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create agent';
        setError(message);
        throw err;
      } finally {
        setIsCreating(false);
      }
    },
    []
  );

  // Refresh KYA token
  const refreshKYAToken = useCallback(async (): Promise<SkyfireKYAToken> => {
    if (!agent) {
      throw new Error('No agent to refresh token for');
    }

    setIsGeneratingToken(true);
    setError(null);

    try {
      const response = await fetch('/api/skyfire/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agentId: agent.id, generateToken: true }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to generate KYA token');
      }

      const token = data.kyaToken as SkyfireKYAToken;
      setKyaToken(token);
      return token;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate token';
      setError(message);
      throw err;
    } finally {
      setIsGeneratingToken(false);
    }
  }, [agent]);

  // Clear the current agent
  const clearAgent = useCallback(() => {
    setAgent(null);
    setKyaToken(null);
    setError(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Check if KYA credentials are valid
  const isKYAValid =
    agent !== null &&
    agent.kyaCredentials.verificationStatus === 'verified' &&
    agent.kyaCredentials.expiresAt > Date.now();

  return {
    agent,
    kyaToken,
    isCreating,
    isGeneratingToken,
    error,
    createAgent,
    refreshKYAToken,
    clearAgent,
    isKYAValid,
  };
}
