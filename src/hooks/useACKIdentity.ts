'use client';

import { useState, useCallback, useEffect } from 'react';
import { useAccount } from 'wagmi';
import {
  createAgentIdentity,
  isIdentityValid,
  serializeIdentity,
  deserializeIdentity,
  type ACKAgentIdentity,
} from '@/lib/ack';

const STORAGE_KEY = 'ack-agent-identity';

interface UseACKIdentityReturn {
  /** Current agent identity */
  identity: ACKAgentIdentity | null;
  /** Whether identity is being created */
  isCreating: boolean;
  /** Whether identity is valid */
  isValid: boolean;
  /** Error message if any */
  error: string | null;
  /** Create a new agent identity */
  createIdentity: (name?: string) => Promise<ACKAgentIdentity>;
  /** Clear the current identity */
  clearIdentity: () => void;
}

/**
 * Hook for managing ACK agent identity
 *
 * Creates and manages a verifiable agent identity derived from the connected wallet
 */
export function useACKIdentity(): UseACKIdentityReturn {
  const { address, isConnected } = useAccount();
  const [identity, setIdentity] = useState<ACKAgentIdentity | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load identity from storage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = deserializeIdentity(stored);
      // Only restore if it matches current wallet
      if (parsed && address && parsed.ownerAddress.toLowerCase() === address.toLowerCase()) {
        if (isIdentityValid(parsed)) {
          setIdentity(parsed);
        } else {
          // Clear expired identity
          localStorage.removeItem(STORAGE_KEY);
        }
      }
    }
  }, [address]);

  // Create a new agent identity
  const createIdentity = useCallback(
    async (name = 'Spending Agent'): Promise<ACKAgentIdentity> => {
      if (!isConnected || !address) {
        throw new Error('Wallet not connected');
      }

      setIsCreating(true);
      setError(null);

      try {
        // Create the identity
        const newIdentity = createAgentIdentity(address, name);

        // Store in localStorage
        if (typeof window !== 'undefined') {
          localStorage.setItem(STORAGE_KEY, serializeIdentity(newIdentity));
        }

        setIdentity(newIdentity);
        return newIdentity;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create identity';
        setError(message);
        throw err;
      } finally {
        setIsCreating(false);
      }
    },
    [address, isConnected]
  );

  // Clear the current identity
  const clearIdentity = useCallback(() => {
    setIdentity(null);
    setError(null);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Check if current identity is valid
  const isValid = identity !== null && isIdentityValid(identity);

  return {
    identity,
    isCreating,
    isValid,
    error,
    createIdentity,
    clearIdentity,
  };
}
