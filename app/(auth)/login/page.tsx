'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading, error, isAuthenticated, clearError } = useAuth();
  
  const [password, setPassword] = useState('');
  const [ip, setIp] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const fetchIP = async () => {
      try {
        const response = await fetch('/api/get-ip');
        const data = await response.json();
        setIp(data.ip);
      } catch (error) {
        console.error('[v0] Error fetching IP:', error);
      }
    };

    fetchIP();
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    clearError();

    if (!password.trim()) {
      setLoginError('Please enter your password');
      return;
    }

    setIsSubmitting(true);

    try {
      await login('Admin121', password, ip);
      router.push('/dashboard');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Login failed. Please try again.';
      setLoginError(errorMsg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Premium Gradient Background */}
      <div className="hidden md:flex md:w-1/3 lg:w-2/5 bg-gradient-to-br from-card-bg to-background border-r border-border flex-col p-8 lg:p-12 relative overflow-hidden">
        {/* Background Glow */}
        <div className="absolute inset-0 glow-purple opacity-20"></div>
        <div className="relative z-10">
          <div className="mb-12">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center mb-4 card-shadow">
              <span className="text-xl font-bold text-white">⚡</span>
            </div>
            <h1 className="text-3xl lg:text-4xl font-bold text-foreground">TradeAI</h1>
            <p className="text-secondary text-sm mt-2">Premium Trading Intelligence</p>
          </div>

          <nav className="space-y-6 mb-12">
            <div>
              <p className="text-xs font-semibold text-secondary uppercase tracking-wider mb-4">Platform</p>
              <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 text-blue-400">
                <span className="text-lg">📊</span>
                <span className="font-medium">Dashboard</span>
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-secondary uppercase tracking-wider mb-4">Features</p>
              <div className="space-y-2">
                <div className="flex items-center gap-3 px-4 py-2.5 text-secondary hover:text-foreground transition-colors cursor-pointer rounded-lg hover:bg-border/30">
                  <span className="text-lg">📈</span>
                  <span className="text-sm font-medium">Live Predictions</span>
                </div>
                <div className="flex items-center gap-3 px-4 py-2.5 text-secondary hover:text-foreground transition-colors cursor-pointer rounded-lg hover:bg-border/30">
                  <span className="text-lg">🎯</span>
                  <span className="text-sm font-medium">Trading Signals</span>
                </div>
              </div>
            </div>
          </nav>

          <div className="mt-auto pt-8 border-t border-border">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-white">🔒</span>
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">Secure Access</p>
                <p className="text-xs text-secondary">Enterprise Grade</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Content - Login Form */}
      <div className="w-full md:w-2/3 lg:w-3/5 flex flex-col items-center justify-center px-6 sm:px-8 lg:px-16 py-12">
        <div className="w-full max-w-md">
          {/* Header */}
          <div className="mb-10">
            <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-3">Welcome Back</h2>
            <p className="text-secondary text-sm lg:text-base">Enter your password to access the trading dashboard</p>
          </div>

          {/* Login Card */}
          <div className="bg-card-bg rounded-xl border border-border card-shadow p-8 sm:p-10">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Password Input */}
              <div>
                <label htmlFor="password" className="block text-sm font-semibold text-foreground mb-3">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isSubmitting || isLoading}
                  placeholder="Enter your password"
                  className="w-full px-4 py-3 rounded-lg border border-border bg-background text-foreground placeholder-secondary/50 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 disabled:opacity-50 disabled:bg-border/20 transition-all"
                  autoComplete="current-password"
                />
              </div>

              {/* IP Info */}
              {ip && (
                <div className="text-xs text-secondary bg-background px-4 py-3 rounded-lg border border-border/50">
                  <p className="font-semibold mb-1">Connected IP</p>
                  <span className="font-mono text-blue-400">{ip}</span>
                </div>
              )}

              {/* Error Message */}
              {(loginError || error) && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                  <p className="text-red-400 text-sm font-medium flex items-center gap-2">
                    <span>⚠️</span>
                    {loginError || error}
                  </p>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting || isLoading || !password}
                className="w-full bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 card-shadow"
              >
                {isSubmitting || isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Verifying...</span>
                  </>
                ) : (
                  <>
                    <span>Access Dashboard</span>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </>
                )}
              </button>
            </form>

            {/* Footer */}
            <div className="mt-8 pt-6 border-t border-border/50 text-center">
              <p className="text-xs text-secondary">
                🔒 Secure access. Your credentials are encrypted and never stored.
              </p>
            </div>
          </div>

          {/* Mobile branding */}
          <div className="md:hidden mt-10 text-center">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center mx-auto mb-4 card-shadow">
              <span className="text-xl font-bold text-white">⚡</span>
            </div>
            <h3 className="text-xl font-bold text-foreground mb-1">TradeAI</h3>
            <p className="text-secondary text-sm">Premium Trading Intelligence Platform</p>
          </div>
        </div>
      </div>
    </div>
  );
}
