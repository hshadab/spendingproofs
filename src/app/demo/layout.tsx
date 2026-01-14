'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ArrowLeft, Shield, Play, CreditCard, AlertTriangle, Wallet, TrendingUp, Bot, Package } from 'lucide-react';

const CORE_DEMOS = [
  { href: '/demo/playground', label: 'Playground', icon: Play, color: 'purple' },
  { href: '/demo/payment', label: 'Payment', icon: CreditCard, color: 'green' },
  { href: '/demo/tamper', label: 'Tamper', icon: AlertTriangle, color: 'amber' },
];

const ENTERPRISE_DEMOS = [
  { href: '/demo/crossmint', label: 'Crossmint', icon: Wallet, color: 'emerald' },
  { href: '/demo/morpho', label: 'Morpho', icon: TrendingUp, color: 'blue' },
  { href: '/demo/openmind', label: 'OpenMind', icon: Bot, color: 'cyan' },
  { href: '/demo/ack', label: 'ACK', icon: Package, color: 'pink' },
];

export default function DemoLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            {/* Left: Back + Logo */}
            <div className="flex items-center gap-4">
              <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
                <ArrowLeft className="w-4 h-4" />
              </Link>
              <Link href="/demo" className="flex items-center gap-2">
                <div className="w-7 h-7 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
                  <Shield className="w-4 h-4" />
                </div>
                <span className="font-semibold text-sm hidden sm:inline">Demos</span>
              </Link>
            </div>

            {/* Center: Demo Links */}
            <div className="flex items-center gap-1">
              {/* Core Demos */}
              <span className="text-[10px] text-gray-600 uppercase tracking-wider mr-1 hidden lg:inline">Core</span>
              {CORE_DEMOS.map(({ href, label, icon: Icon, color }) => {
                const isActive = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      isActive
                        ? `bg-${color}-500/20 text-${color}-400`
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                    style={isActive ? {
                      backgroundColor: `var(--${color}-bg, rgba(139, 92, 246, 0.2))`,
                    } : {}}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? `text-${color}-400` : ''}`} />
                    <span className="hidden md:inline">{label}</span>
                  </Link>
                );
              })}

              {/* Divider */}
              <div className="w-px h-5 bg-gray-700 mx-2" />

              {/* Enterprise Demos */}
              <span className="text-[10px] text-gray-600 uppercase tracking-wider mr-1 hidden lg:inline">Enterprise</span>
              {ENTERPRISE_DEMOS.map(({ href, label, icon: Icon, color }) => {
                const isActive = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      isActive
                        ? `bg-${color}-500/20 text-${color}-400`
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                    style={isActive ? {
                      backgroundColor: `var(--${color}-bg, rgba(139, 92, 246, 0.2))`,
                    } : {}}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? `text-${color}-400` : ''}`} />
                    <span className="hidden md:inline">{label}</span>
                  </Link>
                );
              })}
            </div>

            {/* Right: Main site link */}
            <Link
              href="/"
              className="text-xs text-gray-500 hover:text-white transition-colors hidden sm:block"
            >
              Main Site
            </Link>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {children}
      </main>
    </div>
  );
}
