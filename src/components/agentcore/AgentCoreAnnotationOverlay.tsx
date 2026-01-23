'use client';

import { TrendingUp } from 'lucide-react';

/**
 * Business annotation for executive takeaways
 */
export interface BusinessAnnotation {
  title: string;
  takeaway: string;
  color: 'aws' | 'zkml' | 'combined' | 'enterprise';
  metric?: string;
  metricLabel?: string;
}

// AWS orange color scheme
const ANNOTATION_COLORS = {
  aws: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/50',
    title: 'text-orange-400',
    glow: 'shadow-orange-500/20',
  },
  zkml: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/50',
    title: 'text-yellow-400',
    glow: 'shadow-yellow-500/20',
  },
  combined: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/50',
    title: 'text-green-400',
    glow: 'shadow-green-500/20',
  },
  enterprise: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/50',
    title: 'text-blue-400',
    glow: 'shadow-blue-500/20',
  },
};

interface AgentCoreAnnotationOverlayProps {
  annotation: BusinessAnnotation;
  stepTitle: string;
  onContinue: () => void;
  topOffset?: string; // CSS class for top offset (e.g., "top-16")
}

/**
 * Annotation overlay for AgentCore demo
 * Shows business value metrics between steps
 * Positioned below the playback controls
 */
export function AgentCoreAnnotationOverlay({
  annotation,
  stepTitle,
  onContinue,
  topOffset = "top-0",
}: AgentCoreAnnotationOverlayProps) {
  const colors = ANNOTATION_COLORS[annotation.color];

  return (
    <div
      className={`absolute ${topOffset} left-0 right-0 bottom-0 z-[100] flex items-start justify-center pt-12 bg-black/90 backdrop-blur-sm animate-fadeIn`}
      onClick={onContinue}
    >
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
        .animate-slideDown { animation: slideDown 0.5s ease-out; }
      `}</style>

      <div
        className={`
          max-w-lg mx-4 p-8 rounded-2xl
          ${colors.bg} ${colors.border} border-2
          shadow-2xl ${colors.glow}
          animate-slideDown
        `}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Step context */}
        <div className="text-center mb-4">
          <h3 className="text-sm text-gray-400">{stepTitle}</h3>
        </div>

        {/* Metric - large and centered */}
        {annotation.metric && (
          <div className="flex justify-center mb-6">
            <div className={`px-6 py-4 rounded-xl ${colors.bg} ${colors.border} border-2`}>
              <div className="flex items-center justify-center gap-3">
                <TrendingUp className={`w-6 h-6 ${colors.title}`} />
                <span className={`text-4xl font-bold ${colors.title}`}>{annotation.metric}</span>
              </div>
              <div className="text-sm text-gray-400 text-center mt-1">{annotation.metricLabel}</div>
            </div>
          </div>
        )}

        {/* Title */}
        <h4 className={`text-2xl font-bold ${colors.title} text-center mb-4`}>
          {annotation.title}
        </h4>

        {/* Takeaway text */}
        <p className="text-base text-gray-300 leading-relaxed text-center mb-6">
          {annotation.takeaway}
        </p>

        {/* Continue indicator */}
        <div className="flex flex-col items-center gap-2 pt-4 border-t border-gray-700/50">
          <div className={`w-2 h-2 rounded-full ${colors.title.replace('text-', 'bg-')} animate-pulse`} />
          <span className="text-xs text-gray-500">Click anywhere or wait to continue</span>
        </div>
      </div>
    </div>
  );
}
