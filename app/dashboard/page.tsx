'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { StatsCard } from '@/components/StatsCard'
import { SignalBadge } from '@/components/SignalBadge'
import { GaugeIndicator } from '@/components/GaugeIndicator'
import { AssetSelector } from '@/components/AssetSelector'
import { getMockData } from '@/lib/mockData'
import { DashboardCharts } from '@/components/DashboardCharts'
import { TradingViewWidget } from '@/components/TradingViewWidget'
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

  const handlePredictSignal = async () => {
    setIsLoading(true)
    // Simulate API call to backend predictor
    setTimeout(() => {
      const data = getMockData(selectedAsset)
      setDashboardData(data)
      setLastUpdated(new Date())
      setIsLoading(false)
      // Show success toast (you can add a toast library here)
      console.log('[v0] Prediction signal generated:', data.signal.action)
    }, 1500)
  }

  return (
    <div className="flex bg-slate-50 min-h-screen">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-8 ml-0 md:ml-0">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 mb-2">
                {selectedAsset} Dashboard
              </h1>
              <p className="text-slate-600 text-sm">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </p>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={handlePredictSignal}
                disabled={isLoading}
                className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition flex items-center gap-2"
              >
                <Zap size={18} />
                Predict Signal
              </button>
              <button
                onClick={handleRefresh}
                disabled={isLoading}
                className="p-2 bg-white border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 transition disabled:opacity-50"
              >
                <RefreshCw size={20} className={isLoading ? 'animate-spin' : ''} />
              </button>
              <label className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg cursor-pointer hover:bg-slate-50 transition">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="w-4 h-4 cursor-pointer"
                />
                <span className="text-sm font-medium text-slate-700">Auto-refresh</span>
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

          {/* TradingView Live Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <h3 className="text-slate-900 font-semibold mb-4">XAU/USD Live Chart</h3>
              <TradingViewWidget symbol="XAUUSD" height="400px" />
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <h3 className="text-slate-900 font-semibold mb-4">ETH/USD Live Chart</h3>
              <TradingViewWidget symbol="ETHUSD" height="400px" />
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
            <div className="lg:col-span-1 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-slate-900 font-semibold mb-4">Current Signal</h3>
              <div className="flex flex-col gap-4">
                <SignalBadge signal={dashboardData.signal.action} pips={dashboardData.signal.pips} size="lg" />
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Entry:</span>
                    <span className="text-slate-900 font-mono font-bold">${dashboardData.risk.entry.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Stop Loss:</span>
                    <span className="text-red-600 font-mono font-bold">${dashboardData.risk.stopLoss.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Take Profit:</span>
                    <span className="text-green-600 font-mono font-bold">${dashboardData.risk.takeProfit.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Management Card */}
            <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-slate-900 font-semibold mb-4">Risk Management</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <p className="text-slate-600 text-xs mb-2 font-medium">ENTRY POINT</p>
                  <p className="text-slate-900 font-mono font-bold text-lg">${dashboardData.risk.entry.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-slate-600 text-xs mb-2 font-medium">STOP LOSS</p>
                  <p className="text-red-600 font-mono font-bold text-lg">${dashboardData.risk.stopLoss.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-slate-600 text-xs mb-2 font-medium">TAKE PROFIT</p>
                  <p className="text-green-600 font-mono font-bold text-lg">${dashboardData.risk.takeProfit.toFixed(2)}</p>
                </div>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 flex items-center justify-between">
                <span className="text-slate-600 text-sm">Risk/Reward Ratio</span>
                <span className="text-blue-700 font-bold text-lg">{dashboardData.risk.riskReward}</span>
              </div>
              <p className="text-xs text-slate-600 mt-3">Recommended</p>
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <h3 className="text-slate-900 font-semibold mb-6">Technical Indicators</h3>
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
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <h3 className="text-slate-900 font-semibold mb-4">Prediction History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-300">
                    <th className="text-left py-3 px-4 text-slate-600 font-medium">Date</th>
                    <th className="text-left py-3 px-4 text-slate-600 font-medium">Price</th>
                    <th className="text-left py-3 px-4 text-slate-600 font-medium">Signal</th>
                    <th className="text-left py-3 px-4 text-slate-600 font-medium">Accuracy</th>
                    <th className="text-left py-3 px-4 text-slate-600 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboardData.history.map((entry, idx) => (
                    <tr key={idx} className="border-b border-slate-200 hover:bg-slate-50 transition">
                      <td className="py-3 px-4 text-slate-900">{entry.date}</td>
                      <td className="py-3 px-4 text-slate-900 font-mono">${entry.price.toFixed(2)}</td>
                      <td className="py-3 px-4">
                        <SignalBadge signal={entry.signal as any} size="sm" />
                      </td>
                      <td className="py-3 px-4 text-slate-900 font-mono">{entry.accuracy}%</td>
                      <td className="py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${entry.status === 'Completed' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
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
          <div className="text-center py-8 text-slate-600 text-sm border-t border-slate-300">
            <p>Developed by Muhammad Awais Laal • Educational Project</p>
          </div>
        </div>
      </main>
    </div>
  )
}
