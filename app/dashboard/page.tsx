'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { StatsCard } from '@/components/StatsCard'
import { SignalBadge } from '@/components/SignalBadge'
import { GaugeIndicator } from '@/components/GaugeIndicator'
import { AssetSelector } from '@/components/AssetSelector'
import { getMockData } from '@/lib/mockData'
import { DashboardCharts } from '@/components/DashboardCharts'
import { TrendingUp, TrendingDown, Zap, Target, RefreshCw } from 'lucide-react'
import type { DashboardData } from '@/lib/mockData'

export default function DashboardPage() {
  const [selectedAsset, setSelectedAsset] = useState<'XAU/USD' | 'ETH/USD'>('XAU/USD')
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Load data on mount and when asset changes
  useEffect(() => {
    const loadData = () => {
      setIsLoading(true)
      setTimeout(() => {
        const data = getMockData(selectedAsset)
        setDashboardData(data)
        setLastUpdated(new Date())
        setIsLoading(false)
      }, 300)
    }

    loadData()
  }, [selectedAsset])

  // Auto-refresh data every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      if (dashboardData) {
        const data = getMockData(selectedAsset)
        setDashboardData(data)
        setLastUpdated(new Date())
      }
    }, 30000)

    return () => clearInterval(interval)
  }, [autoRefresh, selectedAsset, dashboardData])

  const handleRefresh = () => {
    setIsLoading(true)
    setTimeout(() => {
      const data = getMockData(selectedAsset)
      setDashboardData(data)
      setLastUpdated(new Date())
      setIsLoading(false)
    }, 300)
  }

  if (!dashboardData) return null

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return 'green'
      case 'SELL':
        return 'red'
      default:
        return 'blue'
    }
  }

  return (
    <div className="flex bg-background min-h-screen">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-8 ml-0 md:ml-0">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-text-primary mb-2">
                {selectedAsset} Dashboard
              </h1>
              <p className="text-text-muted text-sm">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleRefresh}
                disabled={isLoading}
                className="p-2 bg-surface border border-border rounded-lg text-text-primary hover:bg-background transition disabled:opacity-50"
              >
                <RefreshCw size={20} className={isLoading ? 'animate-spin' : ''} />
              </button>
              <label className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg cursor-pointer hover:bg-background transition">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="w-4 h-4 cursor-pointer"
                />
                <span className="text-sm font-medium text-text-primary">Auto-refresh</span>
              </label>
            </div>
          </div>

          {/* Asset Selector */}
          <div className="flex items-center justify-between">
            <AssetSelector selectedAsset={selectedAsset} onAssetChange={setSelectedAsset} />
            <div className="flex items-center gap-2 text-text-muted text-sm md:block hidden">
              <span>Current Price:</span>
              <span className="text-text-primary font-mono font-bold text-lg">
                ${dashboardData.currentPrice.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard
              label="Current Price"
              value={`$${dashboardData.currentPrice.toFixed(2)}`}
              change={dashboardData.changePercent}
              icon={dashboardData.changePercent >= 0 ? <TrendingUp /> : <TrendingDown />}
              color={dashboardData.changePercent >= 0 ? 'green' : 'red'}
            />
            <StatsCard
              label="High / Low"
              value={`$${dashboardData.high.toFixed(2)}`}
              change={((dashboardData.high - dashboardData.low) / dashboardData.low) * 100}
              icon={<Zap />}
              color="blue"
            />
            <StatsCard
              label="Trading Signal"
              value={dashboardData.signal.action}
              icon={<Target />}
              color={getSignalColor(dashboardData.signal.action) as any}
            />
            <StatsCard
              label="Predicted Price (20 min)"
              value={`$${dashboardData.predictedPrice.toFixed(2)}`}
              change={((dashboardData.predictedPrice - dashboardData.currentPrice) / dashboardData.currentPrice) * 100}
              color="purple"
            />
          </div>

          {/* Signal and Risk Management */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Current Signal Card */}
            <div className="lg:col-span-1 bg-surface border border-border rounded-xl p-6">
              <h3 className="text-text-primary font-semibold mb-4">Current Signal</h3>
              <div className="flex flex-col gap-4">
                <SignalBadge signal={dashboardData.signal.action} pips={dashboardData.signal.pips} size="lg" />
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-muted">Entry:</span>
                    <span className="text-text-primary font-mono font-bold">${dashboardData.risk.entry.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Stop Loss:</span>
                    <span className="text-signal-sell font-mono font-bold">${dashboardData.risk.stopLoss.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Take Profit:</span>
                    <span className="text-signal-buy font-mono font-bold">${dashboardData.risk.takeProfit.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Management Card */}
            <div className="lg:col-span-2 bg-surface border border-border rounded-xl p-6">
              <h3 className="text-text-primary font-semibold mb-4">Risk Management</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <p className="text-text-muted text-xs mb-2 font-medium">ENTRY POINT</p>
                  <p className="text-text-primary font-mono font-bold text-lg">${dashboardData.risk.entry.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-text-muted text-xs mb-2 font-medium">STOP LOSS</p>
                  <p className="text-signal-sell font-mono font-bold text-lg">${dashboardData.risk.stopLoss.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-text-muted text-xs mb-2 font-medium">TAKE PROFIT</p>
                  <p className="text-signal-buy font-mono font-bold text-lg">${dashboardData.risk.takeProfit.toFixed(2)}</p>
                </div>
              </div>
              <div className="bg-background border border-accent-primary/30 rounded-lg px-4 py-3 flex items-center justify-between">
                <span className="text-text-muted text-sm">Risk/Reward Ratio</span>
                <span className="text-signal-buy font-bold text-lg">{dashboardData.risk.riskReward}</span>
              </div>
              <p className="text-xs text-text-muted mt-3">Recommended</p>
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <h3 className="text-text-primary font-semibold mb-6">Technical Indicators</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <GaugeIndicator
                label="RSI (Relative Strength Index)"
                value={dashboardData.indicators.rsi.value}
                max={100}
                status={dashboardData.indicators.rsi.status}
                color={dashboardData.indicators.rsi.status === 'Overbought' ? 'red' : dashboardData.indicators.rsi.status === 'Oversold' ? 'green' : 'blue'}
              />
              <GaugeIndicator
                label="MACD (Momentum)"
                value={Math.abs(dashboardData.indicators.macd.value)}
                max={20}
                status={dashboardData.indicators.macd.momentum}
                color={dashboardData.indicators.macd.momentum === 'Bullish' ? 'green' : 'red'}
              />
              <GaugeIndicator
                label="ATR (Volatility)"
                value={dashboardData.indicators.atr.value}
                max={20}
                status={dashboardData.indicators.atr.volatility}
                color={dashboardData.indicators.atr.volatility === 'High' ? 'red' : 'green'}
              />
              <GaugeIndicator
                label="EMA (Trend)"
                value={(dashboardData.indicators.ema.value / dashboardData.currentPrice) * 100 - 100}
                max={10}
                status={dashboardData.indicators.ema.trend}
                color={dashboardData.indicators.ema.trend.includes('Uptrend') ? 'green' : 'red'}
              />
              <GaugeIndicator
                label="ADX (Trend Strength)"
                value={dashboardData.indicators.adx.value}
                max={50}
                status={dashboardData.indicators.adx.strength}
                color={dashboardData.indicators.adx.strength === 'Strong' ? 'green' : 'blue'}
              />
              <GaugeIndicator
                label="CCI (Momentum)"
                value={dashboardData.indicators.cci.value}
                max={200}
                status={dashboardData.indicators.cci.status}
                color={dashboardData.indicators.cci.status === 'Overbought' ? 'red' : 'blue'}
              />
            </div>
          </div>

          {/* Charts */}
          <DashboardCharts data={dashboardData} />

          {/* Prediction History */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <h3 className="text-text-primary font-semibold mb-4">Prediction History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-text-muted font-medium">Date</th>
                    <th className="text-left py-3 px-4 text-text-muted font-medium">Price</th>
                    <th className="text-left py-3 px-4 text-text-muted font-medium">Signal</th>
                    <th className="text-left py-3 px-4 text-text-muted font-medium">Accuracy</th>
                    <th className="text-left py-3 px-4 text-text-muted font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboardData.history.map((entry, idx) => (
                    <tr key={idx} className="border-b border-border/50 hover:bg-background/50 transition">
                      <td className="py-3 px-4 text-text-primary">{entry.date}</td>
                      <td className="py-3 px-4 text-text-primary font-mono">${entry.price.toFixed(2)}</td>
                      <td className="py-3 px-4">
                        <SignalBadge signal={entry.signal as any} size="sm" />
                      </td>
                      <td className="py-3 px-4 text-text-primary font-mono">{entry.accuracy}%</td>
                      <td className="py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${entry.status === 'Completed' ? 'bg-signal-buy/20 text-signal-buy' : 'bg-signal-wait/20 text-signal-wait'}`}>
                          {entry.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center py-8 text-text-muted text-sm border-t border-border">
            <p>Developed by Muhammad Awais Laal • Educational Project</p>
          </div>
        </div>
      </main>
    </div>
  )
}
