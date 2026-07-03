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

  // Fetch client IP
  useEffect(() => {
    const fetchIP = async () => {
      try {
        const response = await fetch('/api/get-ip');
        const data = await response.json();
        setIp(data.ip);
      } catch (error) {
        console.error('[v0] Error fetching IP:', error);
        setLoginError('Unable to determine IP address. Please refresh the page.');
      }
    };

    fetchIP();
  }, []);

  // Redirect if already authenticated
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

    if (!ip) {
      setLoginError('Unable to determine your IP address. Please refresh the page.');
      return;
    }

    setIsSubmitting(true);

    try {
      // Use 'Admin121' as username for password-only login
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
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Logo & Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-lg mb-6 shadow-lg">
            <span className="text-background font-bold text-2xl">$</span>
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-2">Predictor</h1>
          <p className="text-secondary text-sm">XAU/USD & ETH/USD AI Trading Analysis</p>
        </div>

        {/* Login Card */}
        <div className="bg-surface border border-border rounded-xl shadow-2xl p-8 space-y-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-foreground mb-3">
                Access Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isSubmitting || isLoading}
                placeholder="Enter your password"
                className="w-full px-4 py-3 border border-border rounded-lg bg-surface-dark text-foreground placeholder-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 transition-all"
              />
            </div>

            {/* IP Display */}
            <div className="text-xs text-secondary bg-surface-dark px-4 py-3 rounded-lg border border-border">
              <p className="font-semibold mb-2 text-foreground text-opacity-70">Connected from</p>
              <p className="font-mono text-sm text-primary break-all">{ip || 'Loading...'}</p>
            </div>

            {/* Error Messages */}
            {(loginError || error) && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-sm">
                <p className="font-semibold text-red-400 mb-1">Authentication Failed</p>
                <p className="text-red-300 text-xs">{loginError || error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting || isLoading || !ip}
              className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-background font-semibold py-3 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
            >
              {isSubmitting || isLoading ? 'Authenticating...' : 'Access Dashboard'}
            </button>
          </form>
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-xs text-secondary">
          <p className="mb-3">© 2025 Trading Predictor • v2.0</p>
          <p>
            <a href="https://muhammadawaislaal.github.io/My_PortFolio/" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-accent transition-colors">
              Developer
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
