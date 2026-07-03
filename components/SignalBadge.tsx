'use client';

import React from 'react';

interface SignalBadgeProps {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
  pulse?: boolean;
}

export default function SignalBadge({ signal, confidence, pulse = true }: SignalBadgeProps) {
  const signalConfig = {
    BUY: {
      gradient: 'from-green-500 to-emerald-600',
      bg: 'bg-green-500/20',
      text: 'text-green-400',
      border: 'border-green-500/30',
    },
    SELL: {
      gradient: 'from-red-500 to-rose-600',
      bg: 'bg-red-500/20',
      text: 'text-red-400',
      border: 'border-red-500/30',
    },
    HOLD: {
      gradient: 'from-yellow-500 to-amber-600',
      bg: 'bg-yellow-500/20',
      text: 'text-yellow-400',
      border: 'border-yellow-500/30',
    },
  };

  const config = signalConfig[signal];

  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border ${config.bg} ${config.border} ${pulse && signal === 'BUY' ? 'animate-pulse' : ''}`}>
      <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${config.gradient}`}></div>
      <span className={`text-sm font-semibold ${config.text}`}>{signal}</span>
      {confidence && (
        <span className="text-xs text-secondary ml-2">{confidence}%</span>
      )}
    </div>
  );
}
