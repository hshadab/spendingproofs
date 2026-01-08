'use client';

import Link from 'next/link';
import { Play, CreditCard, AlertTriangle, Wallet, TrendingUp, Bot } from 'lucide-react';

const DEMOS = [
  {
    href: '/demo/playground',
    title: 'Playground',
    description: 'Configure policies, simulate purchases, generate proofs',
    icon: Play,
    color: 'purple',
    tag: 'Start Here',
  },
  {
    href: '/demo/payment',
    title: 'Payment Flow',
    description: 'Wallet connect, attestation, USDC payment execution',
    icon: CreditCard,
    color: 'green',
    tag: 'On-Chain',
  },
  {
    href: '/demo/tamper',
    title: 'Tamper Detection',
    description: 'Modify inputs after proof, watch verification fail',
    icon: AlertTriangle,
    color: 'amber',
    tag: 'Security',
  },
  {
    href: '/demo/crossmint',
    title: 'Crossmint',
    description: 'MPC custodial wallets with zkML spending proofs',
    icon: Wallet,
    color: 'emerald',
    tag: 'Enterprise',
  },
  {
    href: '/demo/morpho',
    title: 'Morpho Blue',
    description: 'AI agents managing DeFi vault positions',
    icon: TrendingUp,
    color: 'blue',
    tag: 'DeFi',
  },
  {
    href: '/demo/openmind',
    title: 'OpenMind',
    description: 'Autonomous robot USDC payments via x402',
    icon: Bot,
    color: 'cyan',
    tag: 'Robotics',
  },
];

const COLOR_CLASSES: Record<string, { border: string; bg: string; text: string; iconBg: string }> = {
  purple: {
    border: 'hover:border-purple-500/50',
    bg: 'bg-purple-500/10',
    text: 'text-purple-400',
    iconBg: 'group-hover:bg-purple-500/20',
  },
  green: {
    border: 'hover:border-green-500/50',
    bg: 'bg-green-500/10',
    text: 'text-green-400',
    iconBg: 'group-hover:bg-green-500/20',
  },
  amber: {
    border: 'hover:border-amber-500/50',
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    iconBg: 'group-hover:bg-amber-500/20',
  },
  emerald: {
    border: 'hover:border-emerald-500/50',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-400',
    iconBg: 'group-hover:bg-emerald-500/20',
  },
  blue: {
    border: 'hover:border-blue-500/50',
    bg: 'bg-blue-500/10',
    text: 'text-blue-400',
    iconBg: 'group-hover:bg-blue-500/20',
  },
  cyan: {
    border: 'hover:border-cyan-500/50',
    bg: 'bg-cyan-500/10',
    text: 'text-cyan-400',
    iconBg: 'group-hover:bg-cyan-500/20',
  },
};

export default function DemoHub() {
  return (
    <div className="py-4">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Interactive Demos</h1>
        <p className="text-gray-400">
          Experience zkML spending proofs in action
        </p>
      </div>

      {/* 3x3 Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {DEMOS.map(({ href, title, description, icon: Icon, color, tag }) => {
          const colors = COLOR_CLASSES[color];
          return (
            <Link
              key={href}
              href={href}
              className={`group p-5 bg-[#0d1117] border border-gray-800 rounded-xl ${colors.border} transition-all hover:scale-[1.02]`}
            >
              <div className="flex items-start gap-3">
                <div className={`w-10 h-10 ${colors.bg} rounded-lg flex items-center justify-center ${colors.iconBg} transition-colors`}>
                  <Icon className={`w-5 h-5 ${colors.text}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="font-semibold">{title}</h2>
                    <span className={`text-[10px] ${colors.text} ${colors.bg} px-1.5 py-0.5 rounded`}>
                      {tag}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 line-clamp-2">
                    {description}
                  </p>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Technical Info - Compact */}
      <div className="p-5 bg-[#0d1117] border border-gray-800 rounded-xl">
        <h3 className="font-semibold mb-4 text-sm">Technical Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div>
            <h4 className="text-gray-400 mb-2">Proof System</h4>
            <ul className="space-y-1 text-gray-500">
              <li>JOLT-Atlas SNARK (HyperKZG/BN254)</li>
              <li>~48KB proof size</li>
              <li>2.1s p50 / 3.8s p90 proving</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-400 mb-2">Arc Testnet</h4>
            <ul className="space-y-1 text-gray-500">
              <li>Chain ID: 5042002</li>
              <li>USDC native gas</li>
              <li>Sub-second finality</li>
            </ul>
          </div>
          <div>
            <h4 className="text-gray-400 mb-2">Security</h4>
            <ul className="space-y-1 text-gray-500">
              <li>inputsHash tamper protection</li>
              <li>txIntentHash binding</li>
              <li>Replay protection</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
