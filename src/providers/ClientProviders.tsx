'use client';

import dynamic from 'next/dynamic';
import { type ReactNode } from 'react';

// Dynamically import WalletProvider with SSR disabled to avoid localStorage issues
const WalletProvider = dynamic(
  () => import('@/providers/WalletProvider').then((mod) => mod.WalletProvider),
  { ssr: false }
);

interface ClientProvidersProps {
  children: ReactNode;
}

export function ClientProviders({ children }: ClientProvidersProps) {
  return <WalletProvider>{children}</WalletProvider>;
}
