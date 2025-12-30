'use client';

import { useState } from 'react';
import { Wallet, Cpu, Shield, CheckCircle, Store, ArrowRight, Zap, Lock, FileCheck } from 'lucide-react';

interface DiagramStep {
  id: string;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  detail: string;
  color: string;
  timing?: string;
}

const FLOW_STEPS: DiagramStep[] = [
  {
    id: 'wallet',
    icon: <Wallet className="w-6 h-6" />,
    title: 'Agent Wallet',
    subtitle: 'Policy Config',
    detail: 'Agent has spending policy with limits, allowed categories, and trust thresholds.',
    color: 'purple',
    timing: '0ms',
  },
  {
    id: 'sdk',
    icon: <Cpu className="w-6 h-6" />,
    title: 'Spending SDK',
    subtitle: '8 inputs → Model',
    detail: 'SDK collects transaction context: price, merchant trust, category, urgency, etc.',
    color: 'cyan',
    timing: '~50ms',
  },
  {
    id: 'prover',
    icon: <Shield className="w-6 h-6" />,
    title: 'Jolt Prover',
    subtitle: 'SNARK ~48KB',
    detail: 'HyperKZG proof generated. Cryptographically verifiable decision without revealing inputs.',
    color: 'amber',
    timing: '~2-5s',
  },
  {
    id: 'chain',
    icon: <CheckCircle className="w-6 h-6" />,
    title: 'Arc Chain',
    subtitle: 'Attestation + Payment',
    detail: 'Proof hash attested on-chain. USDC transfer executed with sub-second finality.',
    color: 'green',
    timing: '<1s',
  },
  {
    id: 'merchant',
    icon: <Store className="w-6 h-6" />,
    title: 'Merchant',
    subtitle: 'Verify & Deliver',
    detail: 'Merchant verifies proof attestation, confirms payment, delivers goods/services.',
    color: 'blue',
    timing: 'Instant',
  },
];

interface ArchitectureDiagramProps {
  variant?: 'compact' | 'full' | 'animated';
  showTimings?: boolean;
}

export function ArchitectureDiagram({ variant = 'full', showTimings = true }: ArchitectureDiagramProps) {
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const startAnimation = () => {
    if (isAnimating) return;
    setIsAnimating(true);
    setActiveStep(null);

    FLOW_STEPS.forEach((step, index) => {
      setTimeout(() => {
        setActiveStep(step.id);
        if (index === FLOW_STEPS.length - 1) {
          setTimeout(() => {
            setIsAnimating(false);
          }, 1500);
        }
      }, index * 1000);
    });
  };

  const getColorClasses = (color: string, isActive: boolean) => {
    const colorMap: Record<string, { bg: string; border: string; text: string; glow: string }> = {
      purple: {
        bg: isActive ? 'bg-purple-500/20' : 'bg-purple-500/10',
        border: isActive ? 'border-purple-500' : 'border-purple-500/30',
        text: 'text-purple-400',
        glow: 'shadow-purple-500/30',
      },
      cyan: {
        bg: isActive ? 'bg-cyan-500/20' : 'bg-cyan-500/10',
        border: isActive ? 'border-cyan-500' : 'border-cyan-500/30',
        text: 'text-cyan-400',
        glow: 'shadow-cyan-500/30',
      },
      amber: {
        bg: isActive ? 'bg-amber-500/20' : 'bg-amber-500/10',
        border: isActive ? 'border-amber-500' : 'border-amber-500/30',
        text: 'text-amber-400',
        glow: 'shadow-amber-500/30',
      },
      green: {
        bg: isActive ? 'bg-green-500/20' : 'bg-green-500/10',
        border: isActive ? 'border-green-500' : 'border-green-500/30',
        text: 'text-green-400',
        glow: 'shadow-green-500/30',
      },
      blue: {
        bg: isActive ? 'bg-blue-500/20' : 'bg-blue-500/10',
        border: isActive ? 'border-blue-500' : 'border-blue-500/30',
        text: 'text-blue-400',
        glow: 'shadow-blue-500/30',
      },
    };
    return colorMap[color] || colorMap.purple;
  };

  if (variant === 'compact') {
    return (
      <div className="flex items-center justify-center gap-2 py-4 overflow-x-auto">
        {FLOW_STEPS.map((step, index) => {
          const colors = getColorClasses(step.color, activeStep === step.id);
          return (
            <div key={step.id} className="flex items-center">
              <div
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all ${colors.bg} ${colors.border} ${
                  activeStep === step.id ? `shadow-lg ${colors.glow}` : ''
                }`}
              >
                <div className={colors.text}>{step.icon}</div>
                <span className="text-xs text-gray-300 whitespace-nowrap">{step.title}</span>
              </div>
              {index < FLOW_STEPS.length - 1 && (
                <ArrowRight className="w-4 h-4 text-gray-600 mx-1 flex-shrink-0" />
              )}
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="bg-[#0d1117] border border-gray-800 rounded-xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-400" />
            Spending Proof Flow
          </h3>
          <p className="text-sm text-gray-400">End-to-end agent transaction verification</p>
        </div>
        {variant === 'animated' && (
          <button
            onClick={startAnimation}
            disabled={isAnimating}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
          >
            {isAnimating ? 'Animating...' : 'Play Animation'}
          </button>
        )}
      </div>

      {/* Flow diagram */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        {FLOW_STEPS.map((step, index) => {
          const isActive = activeStep === step.id || (!activeStep && !isAnimating);
          const colors = getColorClasses(step.color, activeStep === step.id);

          return (
            <div key={step.id} className="relative">
              {/* Step card */}
              <div
                className={`p-4 rounded-xl border transition-all duration-300 ${colors.bg} ${colors.border} ${
                  activeStep === step.id ? `shadow-lg ${colors.glow} scale-105` : ''
                } ${!isActive && isAnimating ? 'opacity-40' : ''}`}
                onMouseEnter={() => !isAnimating && setActiveStep(step.id)}
                onMouseLeave={() => !isAnimating && setActiveStep(null)}
              >
                <div className={`${colors.text} mb-2`}>{step.icon}</div>
                <div className="font-medium text-white text-sm mb-1">{step.title}</div>
                <div className="text-xs text-gray-400">{step.subtitle}</div>
                {showTimings && step.timing && (
                  <div className="mt-2 text-xs text-gray-500 flex items-center gap-1">
                    <Lock className="w-3 h-3" />
                    {step.timing}
                  </div>
                )}
              </div>

              {/* Arrow */}
              {index < FLOW_STEPS.length - 1 && (
                <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 z-10">
                  <ArrowRight className={`w-5 h-5 transition-colors ${
                    activeStep === step.id ? 'text-white' : 'text-gray-600'
                  }`} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Detail panel */}
      {activeStep && (
        <div className="p-4 bg-gray-900 rounded-lg">
          <div className="flex items-start gap-3">
            <FileCheck className="w-5 h-5 text-purple-400 mt-0.5" />
            <div>
              <div className="font-medium text-white">
                {FLOW_STEPS.find((s) => s.id === activeStep)?.title}
              </div>
              <p className="text-sm text-gray-400 mt-1">
                {FLOW_STEPS.find((s) => s.id === activeStep)?.detail}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Technical specs */}
      <div className="mt-6 grid grid-cols-4 gap-4">
        <div className="p-3 bg-gray-900 rounded-lg text-center">
          <div className="text-lg font-bold text-purple-400">8</div>
          <div className="text-xs text-gray-500">Model Inputs</div>
        </div>
        <div className="p-3 bg-gray-900 rounded-lg text-center">
          <div className="text-lg font-bold text-cyan-400">3</div>
          <div className="text-xs text-gray-500">Model Outputs</div>
        </div>
        <div className="p-3 bg-gray-900 rounded-lg text-center">
          <div className="text-lg font-bold text-amber-400">~48KB</div>
          <div className="text-xs text-gray-500">Proof Size</div>
        </div>
        <div className="p-3 bg-gray-900 rounded-lg text-center">
          <div className="text-lg font-bold text-green-400">&lt;1s</div>
          <div className="text-xs text-gray-500">Finality</div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-gray-800">
        <div className="text-xs text-gray-500 mb-2">Security Guarantees</div>
        <div className="flex flex-wrap gap-3 text-xs">
          <div className="flex items-center gap-1 text-gray-400">
            <div className="w-2 h-2 bg-purple-500 rounded-full" />
            inputsHash prevents tampering
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <div className="w-2 h-2 bg-cyan-500 rounded-full" />
            txIntentHash binds proof to tx
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <div className="w-2 h-2 bg-amber-500 rounded-full" />
            Nonce prevents replay
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <div className="w-2 h-2 bg-green-500 rounded-full" />
            PolicyRegistry locks model
          </div>
        </div>
      </div>
    </div>
  );
}

// Compact inline version for hero sections
export function ArchitectureFlow() {
  return <ArchitectureDiagram variant="compact" showTimings={false} />;
}
