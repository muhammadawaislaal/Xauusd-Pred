'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import Link from 'next/link';

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading, error, isAuthenticated, clearError } = useAuth();
  
  const [username, setUsername] = useState('');
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

    if (!username.trim()) {
      setLoginError('Please enter your username');
      return;
    }

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
      await login(username, password, ip);
      router.push('/dashboard');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Login failed. Please try again.';
      setLoginError(errorMsg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-secondary/5 flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg mb-4">
            <span className="text-white font-bold text-lg">AI</span>
          </div>
          <h1 className="text-3xl font-bold text-foreground mb-2">AI Predictor</h1>
          <p className="text-secondary text-sm">XAU/USD & ETH/USD Price Predictions</p>
        </div>

        {/* Login Card */}
        <div className="bg-surface border border-border rounded-lg shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Username Field */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-foreground mb-2">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isSubmitting || isLoading}
                placeholder="Enter your username"
                className="w-full px-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50"
              />
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-foreground mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isSubmitting || isLoading}
                placeholder="Enter your password"
                className="w-full px-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50"
              />
            </div>

            {/* IP Display */}
            <div className="text-xs text-secondary bg-background px-3 py-2 rounded border border-border">
              <p className="font-medium mb-1">Your IP Address</p>
              <p className="font-mono text-xs break-all">{ip || 'Loading...'}</p>
            </div>

            {/* Error Messages */}
            {(loginError || error) && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                <p className="font-medium">Login Failed</p>
                <p className="text-xs mt-1">{loginError || error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting || isLoading || !ip}
              className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white font-medium py-3 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting || isLoading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          {/* Demo Info */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-700">
            <p className="font-medium mb-2">Demo Credentials</p>
            <p className="mb-1">Username: <span className="font-mono font-semibold">Admin121</span></p>
            <p>Password: <span className="font-mono font-semibold">Admin121</span></p>
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-xs text-secondary">
          <p className="mb-2">© 2025 XAU/USD & ETH/USD AI Predictor</p>
          <p>
            <a href="https://muhammadawaislaal.github.io/My_PortFolio/" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-accent transition-colors">
              Developer Portfolio
            </a>
            {' • '}
            <a href="mailto:m.awaislaal@gmail.com" className="text-primary hover:text-accent transition-colors">
              Contact
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
