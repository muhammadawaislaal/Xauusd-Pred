'use client';

import React from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { MarketData } from '@/lib/api';

interface ChartDisplayProps {
  data: MarketData | null;
  loading: boolean;
}

export default function ChartDisplay({ data, loading }: ChartDisplayProps) {
  if (loading) {
    return (
      <div className="bg-surface rounded-lg p-6 border border-border shadow-sm animate-pulse">
        <div className="h-96 bg-border rounded"></div>
      </div>
    );
  }

  if (!data || !data.close || data.close.length === 0) {
    return (
      <div className="bg-surface rounded-lg p-6 border border-border shadow-sm text-center">
        <p className="text-secondary">No chart data available</p>
      </div>
    );
  }

  const chartData = data.timestamp.map((ts, idx) => ({
    timestamp: new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    open: data.open[idx],
    high: data.high[idx],
    low: data.low[idx],
    close: data.close[idx],
    volume: Math.round(data.volume[idx] / 1000),
    rsi: data.rsi ? Math.round(data.rsi[idx]) : null,
  }));

  return (
    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
      <h3 className="text-lg font-semibold text-foreground mb-4">Price Action & Technical Analysis</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData}>
          <defs>
            <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#d4a574" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#d4a574" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5ddd3" vertical={false} />
          <XAxis dataKey="timestamp" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5ddd3',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            formatter={(value: any) => {
              if (typeof value === 'number') {
                return value.toFixed(2);
              }
              return value;
            }}
          />
          <Legend />
          <Bar dataKey="volume" yAxisId="right" fill="url(#volumeGradient)" name="Volume" />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="close"
            stroke="#d4a574"
            strokeWidth={2}
            dot={false}
            name="Close Price"
            isAnimationActive={false}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="high"
            stroke="#8b7355"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="High"
            isAnimationActive={false}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="low"
            stroke="#8b7355"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="Low"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs text-secondary mb-1">OPEN</p>
          <p className="text-lg font-semibold text-foreground">${data.open[data.open.length - 1].toFixed(2)}</p>
        </div>
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs text-secondary mb-1">HIGH</p>
          <p className="text-lg font-semibold text-green-600">${Math.max(...data.high).toFixed(2)}</p>
        </div>
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs text-secondary mb-1">LOW</p>
          <p className="text-lg font-semibold text-red-600">${Math.min(...data.low).toFixed(2)}</p>
        </div>
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs text-secondary mb-1">VOLUME</p>
          <p className="text-lg font-semibold text-foreground">{Math.round(data.volume[data.volume.length - 1] / 1000)}K</p>
        </div>
      </div>
    </div>
  );
}
