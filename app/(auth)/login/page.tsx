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
      {/* Left Sidebar */}
      <div className="hidden md:flex md:w-1/3 lg:w-2/5 bg-sidebar-bg text-surface flex-col p-8 lg:p-12">
        <div className="mb-12">
          <div className="w-12 h-12 bg-accent rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-sidebar-bg" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V15a1 1 0 11-2 0V5.414L5.707 10.707a1 1 0 01-1.414-1.414l6-6z" clipRule="evenodd" />
            </svg>
          </div>
          <h1 className="text-2xl lg:text-3xl font-bold">AuxGlobal</h1>
        </div>

        <nav className="space-y-6 mb-12">
          <div>
            <p className="text-xs font-semibold text-surface/70 uppercase tracking-wider mb-4">Main</p>
            <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-accent/20 text-accent">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
              </svg>
              <span className="font-medium">Dashboard</span>
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-surface/70 uppercase tracking-wider mb-4">Trading</p>
            <div className="space-y-2">
              <div className="flex items-center gap-3 px-4 py-2 text-surface/80 hover:text-surface transition-colors cursor-pointer">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7H7v12h6V7z" />
                </svg>
                <span className="text-sm">Analytics</span>
              </div>
              <div className="flex items-center gap-3 px-4 py-2 text-surface/80 hover:text-surface transition-colors cursor-pointer">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="text-sm">Reports</span>
              </div>
            </div>
          </div>
        </nav>

        <div className="mt-auto pt-8 border-t border-surface/10">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-full bg-accent/30 flex items-center justify-center">
              <span className="text-sm font-bold text-accent">A</span>
            </div>
            <div>
              <p className="text-sm font-medium">Trading System</p>
              <p className="text-xs text-surface/70">Premium Access</p>
            </div>
          </div>
        </div>
      </div>

      {/* Right Content */}
      <div className="w-full md:w-2/3 lg:w-3/5 flex flex-col items-center justify-center px-6 sm:px-8 lg:px-16 py-12">
        <div className="w-full max-w-sm">
          {/* Header */}
          <div className="mb-10">
            <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-2">Welcome Back</h2>
            <p className="text-secondary text-sm lg:text-base">Access your premium trading dashboard</p>
          </div>

          {/* Login Card */}
          <div className="bg-surface rounded-2xl border border-border shadow-lg p-8 sm:p-10">
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
                  className="w-full px-4 py-3 rounded-lg border border-border bg-surface-alt text-foreground placeholder-secondary/50 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary disabled:opacity-50 disabled:bg-border/20 transition-all"
                  autoComplete="current-password"
                />
              </div>

              {/* IP Info (subtle) */}
              {ip && (
                <div className="text-xs text-secondary bg-surface-alt px-4 py-2 rounded-lg border border-border/50">
                  <span className="font-mono">{ip}</span>
                </div>
              )}

              {/* Error Message */}
              {(loginError || error) && (
                <div className="bg-error/10 border border-error rounded-lg p-4">
                  <p className="text-error text-sm font-medium">{loginError || error}</p>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting || isLoading || !password}
                className="w-full bg-primary hover:bg-primary-light text-surface font-semibold py-3 px-4 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
                Secure access. Your data is encrypted.
              </p>
            </div>
          </div>

          {/* Mobile branding */}
          <div className="md:hidden mt-8 text-center">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center mx-auto mb-3">
              <svg className="w-5 h-5 text-surface" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V15a1 1 0 11-2 0V5.414L5.707 10.707a1 1 0 01-1.414-1.414l6-6z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="text-secondary text-sm">Trading Intelligence System</p>
          </div>
        </div>
      </div>
    </div>
  );
}
