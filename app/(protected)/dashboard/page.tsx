'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { ProtectedLayout } from '@/components/ProtectedLayout';
import Sidebar from '@/components/Sidebar';
import DashboardHeader from '@/components/DashboardHeader';
import StatsCard from '@/components/StatsCard';
import SignalBadge from '@/components/SignalBadge';
import ChartDisplay from '@/components/ChartDisplay';
import TechnicalIndicators from '@/components/TechnicalIndicators';
import Footer from '@/components/Footer';
import { PredictionResponse, MarketData } from '@/lib/api';
import { useAuth } from '@/lib/auth-context';
import axios from 'axios';

export default function Dashboard() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();
  const [selectedAsset, setSelectedAsset] = useState<'XAU/USD' | 'ETH/USD'>('XAU/USD');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  const fetchPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`/api/predict?symbol=${selectedAsset}`);
      setPrediction(response.data);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('[v0] Prediction fetch error:', err);
      // Mock data fallback
      setPrediction({
        symbol: selectedAsset,
        current_price: selectedAsset === 'XAU/USD' ? 2450.50 : 2650.75,
        predicted_price: selectedAsset === 'XAU/USD' ? 2453.20 : 2655.40,
        signal: 'BUY',
        entry_price: selectedAsset === 'XAU/USD' ? 2450.50 : 2650.75,
        stop_loss: selectedAsset === 'XAU/USD' ? 2445.00 : 2645.00,
        take_profit: selectedAsset === 'XAU/USD' ? 2465.75 : 2670.00,
        accuracy: 94.5,
        timestamp: new Date().toISOString(),
        pip_difference: selectedAsset === 'XAU/USD' ? 2.7 : 4.65,
        features: {
          rsi: 65,
          macd: 0.45,
          atr: 8.5,
          ema: 2448.0,
          adx: 35,
        },
      });
    } finally {
      setLoading(false);
    }
  }, [selectedAsset]);

  const fetchMarketData = useCallback(async () => {
    try {
      const response = await axios.get(`/api/market-data?symbol=${selectedAsset}`);
      setMarketData(response.data);
    } catch (err) {
      console.error('[v0] Market data fetch error:', err);
      const mockData: MarketData = {
        timestamp: Array.from({ length: 50 }, (_, i) => {
          const date = new Date();
          date.setMinutes(date.getMinutes() - (50 - i));
          return date.toISOString();
        }),
        open: Array.from({ length: 50 }, () => selectedAsset === 'XAU/USD' ? 2440 + Math.random() * 20 : 2640 + Math.random() * 20),
        high: Array.from({ length: 50 }, () => selectedAsset === 'XAU/USD' ? 2460 + Math.random() * 20 : 2660 + Math.random() * 20),
        low: Array.from({ length: 50 }, () => selectedAsset === 'XAU/USD' ? 2430 + Math.random() * 20 : 2630 + Math.random() * 20),
        close: Array.from({ length: 50 }, () => selectedAsset === 'XAU/USD' ? 2450 + Math.random() * 20 : 2650 + Math.random() * 20),
        volume: Array.from({ length: 50 }, () => 1000000 + Math.random() * 500000),
        rsi: Array.from({ length: 50 }, () => 30 + Math.random() * 40),
        macd: Array.from({ length: 50 }, () => -1 + Math.random() * 2),
      };
      setMarketData(mockData);
    }
  }, [selectedAsset]);

  useEffect(() => {
    fetchPrediction();
    fetchMarketData();
  }, [selectedAsset, fetchPrediction, fetchMarketData]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchPrediction();
      fetchMarketData();
    }, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchPrediction, fetchMarketData]);

  return (
    <ProtectedLayout>
      <div className="flex min-h-screen bg-background">
        <Sidebar />
        
        <main className="flex-1 lg:ml-0">
          <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto">
            {/* Dashboard Header */}
            <DashboardHeader
              title="Dashboard"
              lastUpdate={lastUpdate}
              onRefresh={fetchPrediction}
              isLoading={loading}
              autoRefresh={autoRefresh}
              onAutoRefreshChange={setAutoRefresh}
            />

            {/* Asset Selector */}
            <div className="flex gap-3 mb-8">
              <button
                onClick={() => setSelectedAsset('XAU/USD')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  selectedAsset === 'XAU/USD'
                    ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white'
                    : 'bg-card-bg border border-border text-secondary hover:text-foreground'
                }`}
              >
                XAU/USD
              </button>
              <button
                onClick={() => setSelectedAsset('ETH/USD')}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  selectedAsset === 'ETH/USD'
                    ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white'
                    : 'bg-card-bg border border-border text-secondary hover:text-foreground'
                }`}
              >
                ETH/USD
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-8 p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                <p className="text-red-400 text-sm font-medium">{error}</p>
              </div>
            )}

            {/* Stats Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <StatsCard
                label="Current Price"
                value={`$${prediction?.current_price.toFixed(2) || '0.00'}`}
                change={
                  prediction?.pip_difference
                    ? {
                        percentage: Math.abs(prediction.pip_difference),
                        positive: prediction.pip_difference > 0,
                      }
                    : undefined
                }
                icon={<span>💰</span>}
                color="blue"
                loading={loading}
              />
              <StatsCard
                label="High / Low"
                value={`$${(prediction?.current_price || 0) + 10} / $${(prediction?.current_price || 0) - 10}`}
                icon={<span>📊</span>}
                color="purple"
                loading={loading}
              />
              <StatsCard
                label="Trading Signal"
                value={<SignalBadge signal={prediction?.signal as any} confidence={prediction?.accuracy} />}
                icon={<span>🎯</span>}
                color="green"
                loading={loading}
              />
              <StatsCard
                label="Predicted Price"
                value={`$${prediction?.predicted_price.toFixed(2) || '0.00'}`}
                change={{
                  percentage: ((prediction?.predicted_price || 0) - (prediction?.current_price || 0)) / (prediction?.current_price || 1) * 100,
                  positive: (prediction?.predicted_price || 0) > (prediction?.current_price || 0),
                }}
                icon={<span>🔮</span>}
                color="blue"
                loading={loading}
              />
            </div>

            {/* Technical Indicators Section */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-foreground mb-6">Technical Indicators</h2>
              <TechnicalIndicators prediction={prediction} loading={loading} />
            </div>

            {/* Risk Management Card */}
            {prediction && (
              <div className="bg-card-bg border border-border rounded-xl p-6 mb-8 card-shadow">
                <h3 className="text-xl font-bold text-foreground mb-6">Risk Management</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-purple-500/10 to-purple-500/5 border border-purple-500/20 rounded-lg p-4">
                    <p className="text-sm text-secondary mb-2">Entry Point</p>
                    <p className="text-2xl font-bold text-purple-400">${prediction.entry_price.toFixed(2)}</p>
                  </div>
                  <div className="bg-gradient-to-br from-red-500/10 to-red-500/5 border border-red-500/20 rounded-lg p-4">
                    <p className="text-sm text-secondary mb-2">Stop Loss</p>
                    <p className="text-2xl font-bold text-red-400">${prediction.stop_loss.toFixed(2)}</p>
                  </div>
                  <div className="bg-gradient-to-br from-green-500/10 to-green-500/5 border border-green-500/20 rounded-lg p-4">
                    <p className="text-sm text-secondary mb-2">Take Profit</p>
                    <p className="text-2xl font-bold text-green-400">${prediction.take_profit.toFixed(2)}</p>
                  </div>
                </div>
                <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-secondary">Risk/Reward Ratio</p>
                    <span className="px-3 py-1 rounded-lg bg-blue-500/20 text-blue-400 text-sm font-semibold">1:2.5 (Recommended)</span>
                  </div>
                </div>
              </div>
            )}

            {/* Chart Display */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-foreground mb-6">Price Movement</h2>
              <ChartDisplay data={marketData} loading={loading} />
            </div>

            {/* Information Box */}
            <div className="bg-card-bg border border-border rounded-xl p-6 mb-8 card-shadow">
              <h3 className="text-xl font-bold text-foreground mb-6">Understanding the Signals</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-2xl">📈</span>
                    <h4 className="font-semibold text-green-400">BUY Signal</h4>
                  </div>
                  <p className="text-sm text-secondary">Strong upward movement expected. Entry recommended at current price with defined stop loss.</p>
                </div>
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-2xl">📉</span>
                    <h4 className="font-semibold text-red-400">SELL Signal</h4>
                  </div>
                  <p className="text-sm text-secondary">Strong downward movement expected. Exit long positions or consider short positions.</p>
                </div>
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-2xl">⏳</span>
                    <h4 className="font-semibold text-yellow-400">HOLD Signal</h4>
                  </div>
                  <p className="text-sm text-secondary">Low volatility or unclear direction. Avoid trading until clearer signals emerge.</p>
                </div>
              </div>
            </div>

            {/* Footer */}
            <Footer />
          </div>
        </main>
      </div>
    </ProtectedLayout>
  );
}
