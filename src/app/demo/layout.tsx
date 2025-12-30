'use client';

import Link from 'next/link';
import { ArrowLeft, Shield } from 'lucide-react';

export default function DemoLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Back to Main</span>
            </Link>
            <div className="hidden md:flex items-center gap-4 text-sm">
              <Link href="/demo/playground" className="text-gray-400 hover:text-white transition-colors">
                Playground
              </Link>
              <Link href="/demo/payment" className="text-gray-400 hover:text-white transition-colors">
                Payment
              </Link>
              <Link href="/demo/tamper" className="text-gray-400 hover:text-white transition-colors">
                Tamper
              </Link>
            </div>
          </div>
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5" />
            </div>
            <span className="font-semibold hidden sm:inline">Spending Proofs</span>
          </Link>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {children}
      </main>
    </div>
  );
}
