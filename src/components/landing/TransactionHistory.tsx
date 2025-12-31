'use client';

import { useState, useEffect } from 'react';
import { ExternalLink, CheckCircle2, Clock, Hash, User } from 'lucide-react';
import { getExplorerUrl, formatAddress, CONTRACTS } from '@/lib/wagmi';

interface ProofEvent {
  proofHash: string;
  submitter: string;
  timestamp: number;
  txHash: string;
  blockNumber: number;
}

// Mock data for demo - in production, fetch from contract events
const mockEvents: ProofEvent[] = [
  {
    proofHash: '0x7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
    submitter: '0x742d35Cc6634C0532925a3b844Bc9e7595f2bD61',
    timestamp: Date.now() - 120000,
    txHash: '0xabc123def456789012345678901234567890123456789012345678901234abcd',
    blockNumber: 1234567,
  },
  {
    proofHash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    submitter: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',
    timestamp: Date.now() - 300000,
    txHash: '0xdef456789012345678901234567890123456789012345678901234567890efgh',
    blockNumber: 1234565,
  },
  {
    proofHash: '0xfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210',
    submitter: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
    timestamp: Date.now() - 600000,
    txHash: '0x123456789012345678901234567890123456789012345678901234567890ijkl',
    blockNumber: 1234560,
  },
];

function formatTimeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function TransactionHistory() {
  const [events, setEvents] = useState<ProofEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setEvents(mockEvents);
      setIsLoading(false);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="bg-[#0a0a0a] border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <h3 className="font-semibold text-sm">Recent Proof Attestations</h3>
        </div>
        <a
          href={getExplorerUrl('address', CONTRACTS.proofAttestation)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
        >
          View all
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>

      {isLoading ? (
        <div className="p-8 text-center">
          <div className="animate-spin w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full mx-auto"></div>
          <p className="text-gray-500 text-sm mt-2">Loading events...</p>
        </div>
      ) : events.length === 0 ? (
        <div className="p-8 text-center">
          <p className="text-gray-500 text-sm">No proof attestations yet</p>
        </div>
      ) : (
        <div className="divide-y divide-gray-800/50">
          {events.map((event, index) => (
            <div
              key={event.txHash}
              className="px-4 py-3 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 min-w-0">
                  <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium">Proof Attested</span>
                      <span className="text-xs text-gray-500 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTimeAgo(event.timestamp)}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
                      <span className="flex items-center gap-1">
                        <Hash className="w-3 h-3" />
                        <code className="text-purple-400">{formatAddress(event.proofHash, 6)}</code>
                      </span>
                      <span className="flex items-center gap-1">
                        <User className="w-3 h-3" />
                        <code>{formatAddress(event.submitter)}</code>
                      </span>
                    </div>
                  </div>
                </div>
                <a
                  href={getExplorerUrl('tx', event.txHash)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-purple-400 transition-colors flex-shrink-0"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="px-4 py-2 bg-[#0d1117]/50 border-t border-gray-800">
        <p className="text-xs text-gray-500 text-center">
          Showing mock data • Connect to Arc Testnet for live events
        </p>
      </div>
    </div>
  );
}
