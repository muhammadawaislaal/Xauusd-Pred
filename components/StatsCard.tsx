'use client';

import React from 'react';

interface StatsCardProps {
  label: string;
  value: string | number;
  change?: {
    percentage: number;
    positive: boolean;
  };
  icon?: React.ReactNode;
  color?: 'purple' | 'blue' | 'green' | 'red';
  loading?: boolean;
}

export default function StatsCard({
  label,
  value,
  change,
  icon,
  color = 'blue',
  loading = false,
}: StatsCardProps) {
  const gradients = {
    purple: 'from-purple-500 to-purple-600',
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    red: 'from-red-500 to-red-600',
  };

  if (loading) {
    return (
      <div className="bg-card-bg border border-border rounded-xl p-6 card-shadow animate-pulse">
        <div className="h-4 bg-border rounded w-24 mb-4"></div>
        <div className="h-8 bg-border rounded w-32"></div>
      </div>
    );
  }

  return (
    <div className="bg-card-bg border border-border rounded-xl p-6 card-shadow hover:border-border-light transition-all duration-300">
      <div className="flex items-start justify-between mb-4">
        <h4 className="text-sm font-medium text-secondary">{label}</h4>
        {icon && (
          <div className={`bg-gradient-to-br ${gradients[color]} p-2.5 rounded-lg`}>
            <div className="text-white opacity-90">{icon}</div>
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-3">
        <div className="text-2xl font-bold text-foreground">{value}</div>
        {change && (
          <div className={`text-xs font-semibold ${change.positive ? 'text-green-400' : 'text-red-400'}`}>
            <span>{change.positive ? '↑' : '↓'} {Math.abs(change.percentage)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}
