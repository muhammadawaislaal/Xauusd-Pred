'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';
import PredictionCard from '@/components/PredictionCard';
import ChartDisplay from '@/components/ChartDisplay';
import TechnicalIndicators from '@/components/TechnicalIndicators';
import Footer from '@/components/Footer';
import Nav from '@/components/Nav';
import { ProtectedLayout } from '@/components/ProtectedLayout';
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

  // Redirect if not authenticated
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
      setError('Failed to fetch prediction. Using mock data.');
      // Fallback to mock data
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
      // Fallback to mock data
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

  // Initial fetch and set up auto-refresh
  useEffect(() => {
    fetchPrediction();
    fetchMarketData();
  }, [selectedAsset, fetchPrediction, fetchMarketData]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchPrediction();
      fetchMarketData();
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, fetchPrediction, fetchMarketData]);

  const handleRunAnalysis = () => {
    fetchPrediction();
    fetchMarketData();
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
          <p className="mt-4 text-secondary">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <ProtectedLayout>
      <div className="min-h-screen bg-background flex flex-col">
        <Nav />
        <Header selectedAsset={selectedAsset} onAssetChange={setSelectedAsset} />

        <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          {/* Control Panel */}
          <div className="bg-surface rounded-lg p-6 border border-border shadow-sm mb-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h2 className="text-xl font-semibold text-foreground mb-2">Analysis Control</h2>
                <p className="text-sm text-secondary">
                  Last updated: {lastUpdate || 'Never'}
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
                <button
                  onClick={handleRunAnalysis}
                  disabled={loading}
                  className={`px-6 py-2 rounded-lg font-semibold smooth-transition ${
                    loading
                      ? 'bg-border text-secondary cursor-not-allowed'
                      : 'bg-primary text-white hover:bg-accent shadow-md'
                  }`}
                >
                  {loading ? '⏳ Analyzing...' : '▶ Run Analysis'}
                </button>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`px-4 py-2 rounded-lg font-semibold smooth-transition border ${
                    autoRefresh
                      ? 'bg-primary text-white border-primary'
                      : 'bg-surface text-foreground border-border'
                  }`}
                >
                  {autoRefresh ? '⏱ Auto-Refresh ON' : '⏸ Auto-Refresh OFF'}
                </button>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
              <p className="text-sm text-yellow-700">⚠ {error}</p>
            </div>
          )}

          {/* Prediction Cards */}
          <PredictionCard prediction={prediction} loading={loading} />

          {/* Technical Indicators */}
          <TechnicalIndicators prediction={prediction} loading={loading} />

          {/* Chart Display */}
          <ChartDisplay data={marketData} loading={loading} />

          {/* Info Box */}
          <div className="bg-primary/10 border border-primary rounded-lg p-6 mt-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">How to Use This Predictor</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Signal Meanings</h4>
                <ul className="space-y-2 text-sm text-secondary">
                  <li className="flex items-start gap-2">
                    <span className="text-green-600 font-bold">📈 BUY</span>
                    <span>Strong upward movement expected (1.5+ pips for Gold, 0.2+ for Ethereum)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 font-bold">📉 SELL</span>
                    <span>Strong downward movement expected</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-amber-600 font-bold">⏳ WAIT</span>
                    <span>Low volatility or unclear direction, avoid trading</span>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-2">Risk Management</h4>
                <ul className="space-y-2 text-sm text-secondary">
                  <li>Always use the provided Stop Loss level to limit losses</li>
                  <li>Take Profit targets offer 1:2.5 risk/reward ratio</li>
                  <li>Standard lot sizes: 0.02 (Gold), 0.10 (Ethereum)</li>
                  <li>Analysis refreshes every 5 minutes with auto-refresh enabled</li>
                </ul>
              </div>
            </div>
          </div>
        </main>

        <Footer />
      </div>
    </ProtectedLayout>
  );
}
