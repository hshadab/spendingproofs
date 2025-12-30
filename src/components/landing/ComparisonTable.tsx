'use client';

import { CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';

interface ComparisonRow {
  dimension: string;
  description: string;
  arc: { value: string; status: 'good' | 'partial' | 'bad' };
  arbitrum: { value: string; status: 'good' | 'partial' | 'bad' };
  optimism: { value: string; status: 'good' | 'partial' | 'bad' };
  base: { value: string; status: 'good' | 'partial' | 'bad' };
}

const COMPARISON_DATA: ComparisonRow[] = [
  {
    dimension: 'Gas Token',
    description: 'Native token used to pay transaction fees',
    arc: { value: 'USDC', status: 'good' },
    arbitrum: { value: 'ETH', status: 'bad' },
    optimism: { value: 'ETH', status: 'bad' },
    base: { value: 'ETH', status: 'bad' },
  },
  {
    dimension: 'Gas Volatility',
    description: 'Price stability of gas payments',
    arc: { value: 'None (stable)', status: 'good' },
    arbitrum: { value: 'High (ETH)', status: 'bad' },
    optimism: { value: 'High (ETH)', status: 'bad' },
    base: { value: 'High (ETH)', status: 'bad' },
  },
  {
    dimension: 'Finality Time',
    description: 'Time until transaction is irreversible',
    arc: { value: '<1 second', status: 'good' },
    arbitrum: { value: '~7 days (soft)', status: 'partial' },
    optimism: { value: '~7 days (soft)', status: 'partial' },
    base: { value: '~7 days (soft)', status: 'partial' },
  },
  {
    dimension: 'Finality Type',
    description: 'Guarantee model for finality',
    arc: { value: 'Deterministic', status: 'good' },
    arbitrum: { value: 'Probabilistic', status: 'partial' },
    optimism: { value: 'Probabilistic', status: 'partial' },
    base: { value: 'Probabilistic', status: 'partial' },
  },
  {
    dimension: 'Reorg Risk',
    description: 'Risk of transaction reversal',
    arc: { value: 'None', status: 'good' },
    arbitrum: { value: 'Sequencer-dependent', status: 'partial' },
    optimism: { value: 'Sequencer-dependent', status: 'partial' },
    base: { value: 'Sequencer-dependent', status: 'partial' },
  },
  {
    dimension: 'Native USDC',
    description: 'Circle-native USDC (not bridged)',
    arc: { value: 'Yes', status: 'good' },
    arbitrum: { value: 'Bridged', status: 'partial' },
    optimism: { value: 'Bridged', status: 'partial' },
    base: { value: 'Yes', status: 'good' },
  },
  {
    dimension: 'Opt-in Privacy',
    description: 'Transaction privacy options',
    arc: { value: 'Supported', status: 'good' },
    arbitrum: { value: 'None', status: 'bad' },
    optimism: { value: 'None', status: 'bad' },
    base: { value: 'None', status: 'bad' },
  },
  {
    dimension: 'zkML Precompile',
    description: 'Native support for zkML verification',
    arc: { value: 'Roadmap', status: 'partial' },
    arbitrum: { value: 'None', status: 'bad' },
    optimism: { value: 'None', status: 'bad' },
    base: { value: 'None', status: 'bad' },
  },
];

const SCENARIO_DATA = [
  {
    scenario: 'ETH price spikes 20%',
    others: "Agent's $1 budget now costs $1.20 in gas value",
    arc: 'Budget unchanged: $1 = $1',
  },
  {
    scenario: 'Agent needs instant confirmation',
    others: 'Wait or accept reorg risk',
    arc: 'Instant deterministic finality',
  },
  {
    scenario: 'Competitor monitors agent',
    others: 'All transactions visible on-chain',
    arc: 'Opt-in confidential transactions',
  },
  {
    scenario: 'Enterprise audit required',
    others: 'Complex reconstruction from logs',
    arc: 'Native attestation trail',
  },
  {
    scenario: 'High-frequency trading agent',
    others: 'Gas estimation failures, stuck txs',
    arc: 'Predictable USDC fees always',
  },
];

function StatusIcon({ status }: { status: 'good' | 'partial' | 'bad' }) {
  switch (status) {
    case 'good':
      return <CheckCircle className="w-4 h-4 text-green-400" />;
    case 'partial':
      return <AlertTriangle className="w-4 h-4 text-amber-400" />;
    case 'bad':
      return <XCircle className="w-4 h-4 text-red-400" />;
  }
}

function getStatusColor(status: 'good' | 'partial' | 'bad'): string {
  switch (status) {
    case 'good':
      return 'text-green-400';
    case 'partial':
      return 'text-amber-400';
    case 'bad':
      return 'text-red-400';
  }
}

interface ComparisonTableProps {
  variant?: 'full' | 'compact';
  showScenarios?: boolean;
}

export function ComparisonTable({ variant = 'full', showScenarios = true }: ComparisonTableProps) {
  return (
    <div className="space-y-8">
      {/* Main Comparison Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="text-left py-3 px-4 font-semibold text-gray-300">Feature</th>
              <th className="text-center py-3 px-4 font-semibold">
                <span className="text-purple-400">Arc</span>
              </th>
              {variant === 'full' && (
                <>
                  <th className="text-center py-3 px-4 font-semibold text-gray-400">Arbitrum</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-400">Optimism</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-400">Base</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {COMPARISON_DATA.map((row, index) => (
              <tr
                key={row.dimension}
                className={`border-b border-gray-800/50 ${
                  index % 2 === 0 ? 'bg-[#0d1117]' : ''
                }`}
              >
                <td className="py-3 px-4">
                  <div className="font-medium text-white">{row.dimension}</div>
                  {variant === 'full' && (
                    <div className="text-xs text-gray-500">{row.description}</div>
                  )}
                </td>
                <td className="text-center py-3 px-4">
                  <div className="flex items-center justify-center gap-2">
                    <StatusIcon status={row.arc.status} />
                    <span className={getStatusColor(row.arc.status)}>{row.arc.value}</span>
                  </div>
                </td>
                {variant === 'full' && (
                  <>
                    <td className="text-center py-3 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <StatusIcon status={row.arbitrum.status} />
                        <span className={getStatusColor(row.arbitrum.status)}>
                          {row.arbitrum.value}
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <StatusIcon status={row.optimism.status} />
                        <span className={getStatusColor(row.optimism.status)}>
                          {row.optimism.value}
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <StatusIcon status={row.base.status} />
                        <span className={getStatusColor(row.base.status)}>{row.base.value}</span>
                      </div>
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Scenario Comparison */}
      {showScenarios && (
        <div className="mt-8">
          <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-cyan-400" />
            What Happens When...
          </h4>
          <div className="grid gap-3">
            {SCENARIO_DATA.map((scenario) => (
              <div
                key={scenario.scenario}
                className="grid md:grid-cols-3 gap-4 p-4 bg-[#0d1117] border border-gray-800 rounded-lg"
              >
                <div>
                  <div className="text-xs text-gray-500 mb-1">Scenario</div>
                  <div className="font-medium text-white">{scenario.scenario}</div>
                </div>
                <div>
                  <div className="text-xs text-red-400 mb-1">On Other L2s</div>
                  <div className="text-sm text-gray-400">{scenario.others}</div>
                </div>
                <div>
                  <div className="text-xs text-green-400 mb-1">On Arc</div>
                  <div className="text-sm text-green-300">{scenario.arc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
        <div className="flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold text-purple-300">Why Arc is Required</div>
            <p className="text-sm text-gray-400 mt-1">
              Autonomous agents require predictable costs (USDC gas), instant finality, and
              privacy options. These aren&apos;t optimizations—they&apos;re requirements for
              agents to operate reliably without human intervention.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export compact version for embedding
export function ComparisonTableCompact() {
  return <ComparisonTable variant="compact" showScenarios={false} />;
}
