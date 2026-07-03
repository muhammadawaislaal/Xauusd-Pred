'use client';

import React from 'react';
import { PredictionResponse } from '@/lib/api';

interface TechnicalIndicatorsProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export default function TechnicalIndicators({ prediction, loading }: TechnicalIndicatorsProps) {
  if (loading || !prediction) {
    return null;
  }

  const features = prediction.features || {
    rsi: 0,
    macd: 0,
    atr: 0,
    ema: 0,
    adx: 0,
  };

  const getIndicatorStatus = (indicator: string, value: number) => {
    if (indicator === 'rsi') {
      if (value > 70) return { label: 'Overbought', color: 'text-red-600' };
      if (value < 30) return { label: 'Oversold', color: 'text-green-600' };
      return { label: 'Neutral', color: 'text-amber-600' };
    }
    if (indicator === 'adx') {
      if (value > 50) return { label: 'Very Strong Trend', color: 'text-green-600' };
      if (value > 40) return { label: 'Strong Trend', color: 'text-blue-600' };
      if (value < 20) return { label: 'Weak Trend', color: 'text-secondary' };
      return { label: 'Moderate Trend', color: 'text-amber-600' };
    }
    return { label: 'Normal', color: 'text-secondary' };
  };

  const indicators = [
    { name: 'RSI', value: features.rsi || 0, max: 100, icon: '📊' },
    { name: 'MACD', value: features.macd || 0, max: 1, icon: '📈' },
    { name: 'ATR', value: features.atr || 0, max: 50, icon: '⚡' },
    { name: 'EMA', value: features.ema || 0, max: 3000, icon: '🔄' },
    { name: 'ADX', value: features.adx || 0, max: 100, icon: '💪' },
  ];

  return (
    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Technical Indicators</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {indicators.map((indicator) => {
          const status = getIndicatorStatus(indicator.name.toLowerCase(), indicator.value);
          const percentage = (indicator.value / indicator.max) * 100;

          return (
            <div key={indicator.name} className="p-4 bg-background rounded border border-border">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xl">{indicator.icon}</span>
                <p className="text-xs font-semibold uppercase text-secondary">{indicator.name}</p>
              </div>
              <p className="text-2xl font-bold text-foreground mb-2">{indicator.value.toFixed(1)}</p>
              <div className="w-full bg-border rounded-full h-2 mb-2">
                <div
                  className="bg-gradient-to-r from-primary to-accent h-2 rounded-full smooth-transition"
                  style={{ width: `${Math.min(percentage, 100)}%` }}
                ></div>
              </div>
              <p className={`text-xs font-medium ${status.color}`}>{status.label}</p>
            </div>
          );
        })}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded border border-blue-200">
        <p className="text-xs text-blue-700">
          <strong>Indicator Guide:</strong> RSI (Overbought/Oversold), MACD (Momentum), ATR (Volatility), EMA (Trend), ADX (Trend Strength)
        </p>
      </div>
    </div>
  );
}
