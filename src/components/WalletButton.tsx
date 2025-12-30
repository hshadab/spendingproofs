'use client';

import { ConnectButton } from '@rainbow-me/rainbowkit';
import { Wallet, ChevronDown, ExternalLink } from 'lucide-react';
import { useWallet } from '@/hooks/useWallet';

interface WalletButtonProps {
  variant?: 'default' | 'compact' | 'full';
  showBalance?: boolean;
  showChain?: boolean;
  className?: string;
}

export function WalletButton({
  variant = 'default',
  showBalance = true,
  showChain = true,
  className = '',
}: WalletButtonProps) {
  if (variant === 'compact') {
    return <CompactWalletButton className={className} />;
  }

  if (variant === 'full') {
    return (
      <FullWalletButton
        showBalance={showBalance}
        showChain={showChain}
        className={className}
      />
    );
  }

  // Default: Use RainbowKit's ConnectButton
  return (
    <ConnectButton
      showBalance={showBalance}
      chainStatus={showChain ? 'icon' : 'none'}
      accountStatus="address"
    />
  );
}

// Compact version for tight spaces
function CompactWalletButton({ className }: { className?: string }) {
  const { isConnected, displayAddress, connect, openAccount } = useWallet();

  if (!isConnected) {
    return (
      <button
        onClick={connect}
        className={`flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-700
                   text-white text-sm font-medium rounded-lg transition-colors ${className}`}
      >
        <Wallet className="w-4 h-4" />
        Connect
      </button>
    );
  }

  return (
    <button
      onClick={openAccount}
      className={`flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700
                 text-white text-sm font-mono rounded-lg transition-colors ${className}`}
    >
      <span className="w-2 h-2 bg-green-400 rounded-full" />
      {displayAddress}
    </button>
  );
}

// Full version with balance display
function FullWalletButton({
  showBalance,
  showChain,
  className,
}: {
  showBalance: boolean;
  showChain: boolean;
  className?: string;
}) {
  const {
    isConnected,
    displayAddress,
    usdcBalanceFormatted,
    isCorrectChain,
    chainName,
    explorerUrl,
    connect,
    openAccount,
    openChain,
  } = useWallet();

  if (!isConnected) {
    return (
      <button
        onClick={connect}
        className={`flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700
                   text-white font-medium rounded-lg transition-colors ${className}`}
      >
        <Wallet className="w-5 h-5" />
        Connect Wallet
      </button>
    );
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Chain indicator */}
      {showChain && (
        <button
          onClick={openChain}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium
                     transition-colors ${
                       isCorrectChain
                         ? 'bg-green-500/10 text-green-400 hover:bg-green-500/20'
                         : 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
                     }`}
        >
          <span className={`w-2 h-2 rounded-full ${isCorrectChain ? 'bg-green-400' : 'bg-red-400'}`} />
          {chainName}
          <ChevronDown className="w-3 h-3" />
        </button>
      )}

      {/* Balance display */}
      {showBalance && usdcBalanceFormatted && (
        <div className="px-3 py-2 bg-gray-800 rounded-lg text-sm">
          <span className="text-gray-400">Balance: </span>
          <span className="font-mono text-white">{usdcBalanceFormatted} USDC</span>
        </div>
      )}

      {/* Account button */}
      <button
        onClick={openAccount}
        className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700
                   rounded-lg transition-colors"
      >
        <span className="w-2 h-2 bg-green-400 rounded-full" />
        <span className="font-mono text-sm">{displayAddress}</span>
        {explorerUrl && (
          <a
            href={explorerUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="text-gray-400 hover:text-white"
          >
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </button>
    </div>
  );
}

// Export a simpler version for use in demos
export function DemoWalletButton() {
  const { isConnected, displayAddress, usdcBalanceFormatted, connect, openAccount } = useWallet();

  if (!isConnected) {
    return (
      <button
        onClick={connect}
        className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700
                   text-white rounded-lg text-sm font-medium transition-colors"
      >
        <Wallet className="w-4 h-4" />
        Connect Wallet
      </button>
    );
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-green-900/30 text-green-400 rounded-lg text-sm">
      <Wallet className="w-4 h-4" />
      <span className="font-mono">{displayAddress}</span>
      {usdcBalanceFormatted && (
        <>
          <span className="text-gray-500">|</span>
          <span>{usdcBalanceFormatted} USDC</span>
        </>
      )}
    </div>
  );
}
