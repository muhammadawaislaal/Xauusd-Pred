'use client';

import React from 'react';

interface HeaderProps {
  selectedAsset: 'XAU/USD' | 'ETH/USD';
  onAssetChange: (asset: 'XAU/USD' | 'ETH/USD') => void;
}

export default function Header({ selectedAsset, onAssetChange }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 bg-surface border-b border-border shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4 gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-surface" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V15a1 1 0 11-2 0V5.414L5.707 10.707a1 1 0 01-1.414-1.414l6-6z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl lg:text-2xl font-bold text-foreground">Dashboard</h1>
              <p className="text-xs lg:text-sm text-secondary">AI Trading Analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={() => onAssetChange('XAU/USD')}
              className={`px-3 sm:px-6 py-2 rounded-lg font-semibold smooth-transition text-sm whitespace-nowrap ${
                selectedAsset === 'XAU/USD'
                  ? 'bg-primary text-surface shadow-lg'
                  : 'bg-surface-alt text-foreground border border-border hover:border-primary'
              }`}
            >
              <span className="hidden sm:inline">Gold (</span>XAU/USD<span className="hidden sm:inline">)</span>
            </button>
            <button
              onClick={() => onAssetChange('ETH/USD')}
              className={`px-3 sm:px-6 py-2 rounded-lg font-semibold smooth-transition text-sm whitespace-nowrap ${
                selectedAsset === 'ETH/USD'
                  ? 'bg-primary text-surface shadow-lg'
                  : 'bg-surface-alt text-foreground border border-border hover:border-primary'
              }`}
            >
              <span className="hidden sm:inline">Ethereum (</span>ETH/USD<span className="hidden sm:inline">)</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
