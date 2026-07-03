'use client';

import React from 'react';

interface DashboardHeaderProps {
  title: string;
  lastUpdate?: string;
  onRefresh?: () => void;
  isLoading?: boolean;
  autoRefresh?: boolean;
  onAutoRefreshChange?: (enabled: boolean) => void;
}

export default function DashboardHeader({
  title,
  lastUpdate,
  onRefresh,
  isLoading = false,
  autoRefresh = true,
  onAutoRefreshChange,
}: DashboardHeaderProps) {
  return (
    <div className="bg-card-bg border border-border rounded-xl p-6 mb-8 card-shadow">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2">{title}</h1>
          {lastUpdate && (
            <p className="text-sm text-secondary">
              Last updated: <span className="font-mono text-blue-400">{lastUpdate}</span>
            </p>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-purple-500 to-blue-500 text-white font-medium text-sm hover:from-purple-600 hover:to-blue-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Refresh</span>
              </>
            )}
          </button>

          {onAutoRefreshChange && (
            <button
              onClick={() => onAutoRefreshChange(!autoRefresh)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border transition-all font-medium text-sm ${
                autoRefresh
                  ? 'bg-gradient-to-r from-purple-500/20 to-blue-500/20 border-purple-500/30 text-blue-400'
                  : 'bg-border/30 border-border text-secondary hover:text-foreground'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{autoRefresh ? 'Auto ON' : 'Auto OFF'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
