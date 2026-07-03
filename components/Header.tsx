'use client';

import React from 'react';

interface HeaderProps {
  selectedAsset: 'XAU/USD' | 'ETH/USD';
  onAssetChange: (asset: 'XAU/USD' | 'ETH/USD') => void;
}

export default function Header({ selectedAsset, onAssetChange }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 glass shadow-sm border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">₹</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">AI Predictor</h1>
              <p className="text-sm text-secondary">Professional Trading Signals</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => onAssetChange('XAU/USD')}
              className={`px-6 py-2 rounded-lg font-semibold smooth-transition ${
                selectedAsset === 'XAU/USD'
                  ? 'bg-primary text-white shadow-md'
                  : 'bg-surface text-foreground border border-border hover:bg-border'
              }`}
            >
              Gold (XAU/USD)
            </button>
            <button
              onClick={() => onAssetChange('ETH/USD')}
              className={`px-6 py-2 rounded-lg font-semibold smooth-transition ${
                selectedAsset === 'ETH/USD'
                  ? 'bg-primary text-white shadow-md'
                  : 'bg-surface text-foreground border border-border hover:bg-border'
              }`}
            >
              Ethereum (ETH/USD)
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
