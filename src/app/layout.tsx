import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ClientProviders } from '@/providers/ClientProviders';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Arc Spending Policy Proofs',
  description: 'Jolt-Atlas zkML spending policy proofs for autonomous agents on Arc chain',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#0a0a0a] text-white`}>
        <ClientProviders>
          {children}
        </ClientProviders>
      </body>
    </html>
  );
}
