'use client';

import React from 'react';
import { PredictionResponse } from '@/lib/api';

interface PredictionCardProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export default function PredictionCard({ prediction, loading }: PredictionCardProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <div className="bg-surface rounded-lg p-6 border border-border animate-pulse">
          <div className="h-6 bg-border rounded w-1/3 mb-4"></div>
          <div className="h-8 bg-border rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-border rounded w-2/3"></div>
        </div>
        <div className="bg-surface rounded-lg p-6 border border-border animate-pulse">
          <div className="h-6 bg-border rounded w-1/3 mb-4"></div>
          <div className="h-8 bg-border rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-border rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-surface rounded-lg p-6 border border-border text-center">
        <p className="text-secondary">No prediction data available. Please run analysis.</p>
      </div>
    );
  }

  const isPositiveSignal = prediction.signal === 'BUY';
  const isNegativeSignal = prediction.signal === 'SELL';
  const isNeutral = prediction.signal === 'WAIT';

  const signalColor = isPositiveSignal ? 'text-green-600 bg-green-50' : isNegativeSignal ? 'text-red-600 bg-red-50' : 'text-amber-600 bg-amber-50';
  const signalBorder = isPositiveSignal ? 'border-green-200' : isNegativeSignal ? 'border-red-200' : 'border-amber-200';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      {/* Current Price Card */}
      <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
        <p className="text-sm text-secondary font-semibold mb-2 uppercase tracking-wide">Current Price</p>
        <div className="flex items-baseline gap-2 mb-4">
          <h2 className="text-4xl font-bold text-foreground">${prediction.current_price.toFixed(2)}</h2>
          <span className="text-sm text-secondary">{prediction.symbol}</span>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-secondary mb-1">HIGH (Today)</p>
            <p className="text-lg font-semibold text-foreground">$2,456.80</p>
          </div>
          <div>
            <p className="text-xs text-secondary mb-1">LOW (Today)</p>
            <p className="text-lg font-semibold text-foreground">$2,445.20</p>
          </div>
        </div>
      </div>

      {/* Prediction Card */}
      <div className={`bg-surface rounded-lg p-6 border-2 shadow-sm ${signalBorder}`}>
        <p className="text-sm text-secondary font-semibold mb-2 uppercase tracking-wide">Predicted Price (20 min)</p>
        <div className="flex items-baseline gap-2 mb-4">
          <h2 className="text-4xl font-bold text-foreground">${prediction.predicted_price.toFixed(2)}</h2>
          <span className={`text-sm font-bold ${prediction.pip_difference > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {prediction.pip_difference > 0 ? '↑' : '↓'} {Math.abs(prediction.pip_difference).toFixed(2)} pips
          </span>
        </div>
        <div className="mb-4">
          <p className="text-xs text-secondary mb-2">Confidence Level</p>
          <div className="w-full bg-border rounded-full h-2">
            <div
              className="bg-gradient-to-r from-primary to-accent h-2 rounded-full"
              style={{ width: `${prediction.accuracy}%` }}
            ></div>
          </div>
          <p className="text-xs text-secondary mt-1">{prediction.accuracy.toFixed(0)}% Accuracy</p>
        </div>
      </div>

      {/* Signal Card */}
      <div className={`rounded-lg p-6 border-2 shadow-sm ${signalColor} ${signalBorder}`}>
        <p className="text-xs font-semibold uppercase tracking-wider mb-3">Trading Signal</p>
        <div className="text-center">
          <p className={`text-5xl font-bold mb-2 ${isPositiveSignal ? 'text-green-600' : isNegativeSignal ? 'text-red-600' : 'text-amber-600'}`}>
            {prediction.signal}
          </p>
          {prediction.signal !== 'WAIT' && (
            <p className="text-sm font-medium">
              {isPositiveSignal ? 'Strong upward movement expected' : 'Strong downward movement expected'}
            </p>
          )}
          {prediction.signal === 'WAIT' && (
            <p className="text-sm font-medium">
              Low volatility or unclear direction
            </p>
          )}
        </div>
      </div>

      {/* Risk Management Card */}
      {prediction.signal !== 'WAIT' && (
        <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
          <p className="text-sm text-secondary font-semibold mb-4 uppercase tracking-wide">Risk Management</p>
          <div className="space-y-3">
            <div className="flex justify-between items-center pb-3 border-b border-border">
              <span className="text-sm text-secondary">Entry Point</span>
              <span className="font-semibold text-foreground">${prediction.entry_price?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-border">
              <span className="text-sm text-secondary">Stop Loss</span>
              <span className="font-semibold text-red-600">${prediction.stop_loss?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary">Take Profit</span>
              <span className="font-semibold text-green-600">${prediction.take_profit?.toFixed(2)}</span>
            </div>
          </div>
          <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
            <p className="text-xs text-blue-700">
              <strong>Risk/Reward Ratio:</strong> 1:2.5 (Recommended)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
