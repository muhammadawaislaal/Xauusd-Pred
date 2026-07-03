'use client'

import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart } from 'recharts'
import type { DashboardData } from '@/lib/mockData'

interface DashboardChartsProps {
  data: DashboardData
}

export function DashboardCharts({ data }: DashboardChartsProps) {
  // Combine historical and predicted data for forecast chart
  const forecastChartData = [
    ...data.forecastData.historical,
    ...data.forecastData.predicted,
  ]

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white border border-slate-300 rounded-lg p-3 shadow-lg">
          <p className="text-slate-900 text-sm font-mono">${payload[0].value.toFixed(2)}</p>
          <p className="text-slate-600 text-xs">{payload[0].payload.time}</p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-4">
      {/* Candlestick Chart */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-slate-900 font-semibold mb-4">Price Action (5-min candlesticks)</h3>
        <div className="w-full h-80 flex items-center justify-center">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.candlestickData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                dataKey="close"
                fill="#22c55e"
                radius={[4, 4, 0, 0]}
                isAnimationActive={true}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecast Chart */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-slate-900 font-semibold mb-4">Price Forecast (20-min prediction)</h3>
        <div className="w-full h-80 flex items-center justify-center relative">
          {/* Watermark */}
          <div className="absolute bottom-4 right-4 text-slate-300 text-xs font-semibold pointer-events-none">
            Awais Trading Aala
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={forecastChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip content={<CustomTooltip />} />
              {/* Historical data line */}
              <Line
                type="monotone"
                dataKey="price"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
                name="Historical"
              />
              {/* Predicted data line */}
              <Line
                type="monotone"
                dataKey="price"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ fill: '#f97316', r: 4 }}
                isAnimationActive={true}
                name="Predicted"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 flex items-center gap-6 text-sm flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-accent-secondary"></div>
            <span className="text-text-muted">Historical Price</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-orange-500 opacity-60" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #f97316 0, #f97316 5px, transparent 5px, transparent 10px)' }}></div>
            <span className="text-text-muted">Predicted Price</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-signal-buy"></div>
            <span className="text-text-muted">Entry Point</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-signal-sell"></div>
            <span className="text-text-muted">Stop Loss</span>
          </div>
        </div>
      </div>

      {/* TradingView Widget */}
      <div className="bg-surface border border-border rounded-xl p-6">
        <h3 className="text-text-primary font-semibold mb-4">TradingView Chart</h3>
        <div className="bg-background rounded-lg p-8 text-center">
          <div className="inline-flex flex-col items-center gap-3 text-text-muted">
            <div className="text-4xl">📊</div>
            <p className="text-sm font-medium">TradingView Chart Widget</p>
            <p className="text-xs">
              {data.asset === 'XAU/USD' ? 'OANDA:XAUUSD' : 'BINANCE:ETHUSDT'} • 5-minute interval
            </p>
            <p className="text-xs max-w-xs text-text-muted/60">
              Embed TradingView iframe with real-time price charts. Requires TradingView license.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
