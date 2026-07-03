'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { StatsCard } from '@/components/StatsCard'
import { SignalBadge } from '@/components/SignalBadge'
import { GaugeIndicator } from '@/components/GaugeIndicator'
import { AssetSelector } from '@/components/AssetSelector'
import { DashboardCharts } from '@/components/DashboardCharts'
import { TradingViewWidget } from '@/components/TradingViewWidget'
import { StatsCard as BaseStatsCard } from '@/components/StatsCard'
import { SignalBadge as BaseSignalBadge } from '@/components/SignalBadge'
import { getPredictedSignal } from '@/lib/api'
import { TrendingUp, TrendingDown, Zap, Target, RefreshCw } from 'lucide-react'
import type { SignalResponse } from '@/lib/api'

export default function DashboardPage() {
  const [selectedAsset, setSelectedAsset] = useState<'XAU/USD' | 'ETH/USD'>('XAU/USD')
  const [dashboardData, setDashboardData] = useState<SignalResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Load real market data from backend API only
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true)
      try {
        const signalData = await getPredictedSignal(selectedAsset)
        
        if (signalData) {
          setDashboardData(signalData as any)
          console.log('[v0] Professional market data loaded from backend:', selectedAsset)
        } else {
          // Show error state instead of mock data
          setDashboardData(null)
          console.error('[v0] Unable to load market data - backend API unavailable')
        }
      } catch (error) {
        console.error('[v0] Critical error loading market data:', error)
        setDashboardData(null)
      } finally {
        setIsLoading(false)
      }
    }

    loadData()
  }, [selectedAsset])

  // Auto-refresh real data every 30 seconds (no mock fallback)
  useEffect(() => {
    if (!autoRefresh || !dashboardData) return

    const interval = setInterval(async () => {
      try {
        const signalData = await getPredictedSignal(selectedAsset)
        
        if (signalData) {
          setDashboardData(signalData as any)
          setLastUpdated(new Date())
          console.log('[v0] Auto-refresh: Real market data updated')
        } else {
          console.warn('[v0] Auto-refresh: Could not fetch updated data from backend')
        }
      } catch (error) {
        console.error('[v0] Auto-refresh error:', error)
      }
    }, 30000)

    return () => clearInterval(interval)
  }, [autoRefresh, selectedAsset, dashboardData])

  const handlePredictSignal = async () => {
    setIsLoading(true)
    try {
      const signalData = await getPredictedSignal(selectedAsset)
      
      if (signalData) {
        setDashboardData(signalData as any)
        setLastUpdated(new Date())
        console.log('[v0] Predict Signal: New analysis from backend AI model')
      } else {
        console.error('[v0] Unable to generate prediction - backend unavailable')
      }
    } catch (error) {
      console.error('[v0] Prediction generation failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Show error state if backend is unavailable
  if (!dashboardData) {
    return (
      <div className="flex bg-slate-50 min-h-screen">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="bg-white border border-red-200 rounded-xl p-8 max-w-md text-center">
            <h2 className="text-xl font-bold text-slate-900 mb-2">Backend Unavailable</h2>
            <p className="text-slate-600 mb-4">
              Unable to connect to the backend API. Please ensure the backend server is running and NEXT_PUBLIC_API_URL is configured correctly.
            </p>
            <p className="text-sm text-slate-500 font-mono">
              Expected: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}
            </p>
          </div>
        </main>
      </div>
    )
  }

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


        const updatedData = {
          ...dashboardData,
          currentPrice: signalData.price,
          signal: {
            action: signalData.signal as 'BUY' | 'SELL' | 'WAIT',
            confidence: signalData.confidence,
            timestamp: signalData.timestamp,
          },
          technicalIndicators: signalData.technicalIndicators,
          risk: signalData.riskManagement,
        }
        setDashboardData(updatedData)
        console.log('[v0] Real-time signal from backend:', signalData.signal)
      } else {
        // Fallback to mock data if API fails
        const data = getMockData(selectedAsset)
        setDashboardData(data)
        console.log('[v0] Using fallback mock data - backend API unavailable')
      }
      
      setLastUpdated(new Date())
    } catch (error) {
      console.error('[v0] Error fetching prediction:', error)
      // Fallback to mock data on error
      const data = getMockData(selectedAsset)
      setDashboardData(data)
    } finally {
      setIsLoading(false)
    }
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

          {/* Live TradingView Chart - Synchronized with real market data */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <TradingViewWidget 
              symbol={selectedAsset === 'XAU/USD' ? 'XAUUSD' : 'ETHUSD'} 
              height="500px" 
            />
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard
              label="Current Price"
              value={`${dashboardData.currentPrice.toFixed(2)}`}
              change={dashboardData.changePercent}
              icon={dashboardData.changePercent >= 0 ? <TrendingUp /> : <TrendingDown />}
              color={dashboardData.changePercent >= 0 ? 'green' : 'red'}
            />
            <StatsCard
              label="High / Low"
              value={`${dashboardData.high.toFixed(2)}`}
              change={((dashboardData.high - dashboardData.low) / dashboardData.low) * 100}
              icon={<Zap />}
              color="blue"
            />
            <StatsCard
              label="Trading Signal"
              value={dashboardData.signal.action}
              change={dashboardData.signal.pips}
              icon={<Target />}
              color={getSignalColor(dashboardData.signal.action)}
            />
            <StatsCard
              label="Predicted Price (20 min)"
              value={`${dashboardData.predictedPrice.toFixed(2)}`}
              change={((dashboardData.predictedPrice - dashboardData.currentPrice) / dashboardData.currentPrice) * 100}
              icon={<TrendingUp />}
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
                    <span className="text-slate-900 font-mono font-bold">{dashboardData.risk.entry.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Stop Loss:</span>
                    <span className="text-red-600 font-mono font-bold">{dashboardData.risk.stopLoss.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Take Profit:</span>
                    <span className="text-green-600 font-mono font-bold">{dashboardData.risk.takeProfit.toFixed(2)}</span>
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
                  <p className="text-slate-900 font-mono font-bold text-lg">{dashboardData.risk.entry.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-slate-600 text-xs mb-2 font-medium">STOP LOSS</p>
                  <p className="text-red-600 font-mono font-bold text-lg">{dashboardData.risk.stopLoss.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-slate-600 text-xs mb-2 font-medium">TAKE PROFIT</p>
                  <p className="text-green-600 font-mono font-bold text-lg">{dashboardData.risk.takeProfit.toFixed(2)}</p>
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
                      <td className="py-3 px-4 text-slate-900 font-mono">{entry.price.toFixed(2)}</td>
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
            <p>
              Developed by{' '}
              <a
                href="https://muhammadawaislaal.github.io/My_PortFolio/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-700 underline hover:underline font-semibold transition"
              >
                Muhammad Awais Laal
              </a>
              {' '}• Educational Project
            </p>
          </div>
        </div>
      </main>
      
      {/* Live Chat Support */}
      <LiveChat />
    </div>
  )
}
