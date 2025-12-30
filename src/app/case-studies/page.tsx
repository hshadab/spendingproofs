'use client';

import Link from 'next/link';
import { ArrowLeft, ArrowRight, Cpu, ShoppingCart, Briefcase, Bot } from 'lucide-react';

const CASE_STUDIES = [
  {
    id: 'compute-agent',
    title: 'Agent X Purchases Compute',
    description: 'An AI agent autonomously procures GPU compute from a marketplace, verified by spending proofs.',
    icon: <Cpu className="w-6 h-6" />,
    color: 'purple',
    metrics: {
      proofTime: '2.3s',
      txCost: '$0.02',
      finality: '<1s',
    },
    href: '/case-studies/compute-agent',
  },
  {
    id: 'shopping-agent',
    title: 'Shopping Agent',
    description: 'E-commerce agent handles inventory restocking within defined budget constraints.',
    icon: <ShoppingCart className="w-6 h-6" />,
    color: 'cyan',
    metrics: {
      proofTime: '1.8s',
      txCost: '$0.01',
      finality: '<1s',
    },
    href: '/case-studies/shopping-agent',
    comingSoon: true,
  },
  {
    id: 'treasury-agent',
    title: 'Treasury Management',
    description: 'Corporate treasury agent manages payments within multi-signature approval flows.',
    icon: <Briefcase className="w-6 h-6" />,
    color: 'amber',
    metrics: {
      proofTime: '3.1s',
      txCost: '$0.03',
      finality: '<1s',
    },
    href: '/case-studies/treasury-agent',
    comingSoon: true,
  },
];

export default function CaseStudiesPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <div className="border-b border-gray-800 bg-[#0d1117]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center">
              <Bot className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Case Studies</h1>
              <p className="text-gray-400">
                Real-world scenarios where spending proofs enable autonomous agent commerce
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {CASE_STUDIES.map((study) => {
            const colorClasses = {
              purple: 'border-purple-500/30 hover:border-purple-500',
              cyan: 'border-cyan-500/30 hover:border-cyan-500',
              amber: 'border-amber-500/30 hover:border-amber-500',
            };
            const iconColors = {
              purple: 'bg-purple-500/10 text-purple-400',
              cyan: 'bg-cyan-500/10 text-cyan-400',
              amber: 'bg-amber-500/10 text-amber-400',
            };

            return (
              <div
                key={study.id}
                className={`bg-[#0d1117] border rounded-xl p-6 transition-all ${colorClasses[study.color as keyof typeof colorClasses]} ${
                  study.comingSoon ? 'opacity-60' : ''
                }`}
              >
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${iconColors[study.color as keyof typeof iconColors]}`}>
                  {study.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{study.title}</h3>
                <p className="text-sm text-gray-400 mb-4">{study.description}</p>

                <div className="grid grid-cols-3 gap-2 mb-4">
                  <div className="text-center p-2 bg-gray-900 rounded">
                    <div className="text-xs text-gray-500">Proof</div>
                    <div className="text-sm font-medium">{study.metrics.proofTime}</div>
                  </div>
                  <div className="text-center p-2 bg-gray-900 rounded">
                    <div className="text-xs text-gray-500">Cost</div>
                    <div className="text-sm font-medium">{study.metrics.txCost}</div>
                  </div>
                  <div className="text-center p-2 bg-gray-900 rounded">
                    <div className="text-xs text-gray-500">Finality</div>
                    <div className="text-sm font-medium">{study.metrics.finality}</div>
                  </div>
                </div>

                {study.comingSoon ? (
                  <span className="inline-block px-3 py-1.5 text-sm text-gray-500 bg-gray-800 rounded-lg">
                    Coming Soon
                  </span>
                ) : (
                  <Link
                    href={study.href}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-colors"
                  >
                    View Case Study
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
