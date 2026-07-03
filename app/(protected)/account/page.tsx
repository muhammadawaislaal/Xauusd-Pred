'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { ProtectedLayout } from '@/components/ProtectedLayout';
import Nav from '@/components/Nav';
import Footer from '@/components/Footer';

export default function AccountPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
          <p className="mt-4 text-secondary">Loading account...</p>
        </div>
      </div>
    );
  }

  return (
    <ProtectedLayout>
      <div className="min-h-screen bg-background flex flex-col">
        <Nav />
        <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-2">Account Settings</h1>
            <p className="text-secondary">Manage your account information and subscription details</p>
          </div>

          {/* Account Info Card */}
          <div className="bg-surface border border-border rounded-lg shadow-sm p-6 mb-6">
            <h2 className="text-xl font-semibold text-foreground mb-6">Account Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Left Column */}
              <div className="space-y-6">
                {/* Username */}
                <div>
                  <label className="block text-sm font-medium text-secondary mb-2">Username</label>
                  <div className="px-4 py-3 bg-background rounded border border-border">
                    <p className="font-mono text-foreground font-semibold">{user?.username}</p>
                  </div>
                </div>

                {/* IP Address */}
                <div>
                  <label className="block text-sm font-medium text-secondary mb-2">Registered IP Address</label>
                  <div className="px-4 py-3 bg-background rounded border border-border">
                    <p className="font-mono text-foreground font-semibold break-all">{user?.ip}</p>
                  </div>
                </div>
              </div>

              {/* Right Column - Subscription Info */}
              <div>
                <div className="bg-primary/10 border border-primary rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4">Subscription Status</h3>
                  
                  {user?.subscription ? (
                    <div className="space-y-4">
                      {/* Status */}
                      <div>
                        <p className="text-sm text-secondary mb-1">Status</p>
                        <div className="flex items-center gap-2">
                          <div className={`h-3 w-3 rounded-full ${user.subscription.status === 'active' ? 'bg-green-500' : 'bg-red-500'}`}></div>
                          <span className="font-semibold text-foreground capitalize">{user.subscription.status}</span>
                        </div>
                      </div>

                      {/* Plan */}
                      {user.subscription.plan && (
                        <div>
                          <p className="text-sm text-secondary mb-1">Plan</p>
                          <p className="font-semibold text-foreground capitalize">{user.subscription.plan}</p>
                        </div>
                      )}

                      {/* Expiry Date */}
                      {user.subscription.expiryDate && (
                        <div>
                          <p className="text-sm text-secondary mb-1">Expiry Date</p>
                          <p className="font-semibold text-foreground">
                            {new Date(user.subscription.expiryDate).toLocaleDateString('en-US', {
                              year: 'numeric',
                              month: 'long',
                              day: 'numeric',
                            })}
                          </p>
                          {getDaysUntilExpiry(user.subscription.expiryDate) <= 30 && (
                            <p className="text-xs text-amber-600 mt-2 font-medium">
                              ⚠ Expires in {getDaysUntilExpiry(user.subscription.expiryDate)} days
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-secondary">No subscription information available</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Features Card */}
          <div className="bg-surface border border-border rounded-lg shadow-sm p-6 mb-6">
            <h2 className="text-xl font-semibold text-foreground mb-6">Active Features</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <FeatureItem title="Real-time Predictions" description="Live AI-powered price predictions" />
              <FeatureItem title="Trading Signals" description="BUY/SELL/WAIT signals with confidence levels" />
              <FeatureItem title="Technical Analysis" description="5 technical indicators (RSI, MACD, ATR, EMA, ADX)" />
              <FeatureItem title="Risk Management" description="Stop Loss and Take Profit calculations" />
              <FeatureItem title="Multi-Asset Support" description="Trade XAU/USD and ETH/USD" />
              <FeatureItem title="Auto-Refresh" description="5-minute automatic data updates" />
            </div>
          </div>

          {/* Security Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-blue-900 mb-3">Security Notice</h2>
            <ul className="space-y-2 text-sm text-blue-800">
              <li className="flex items-start gap-3">
                <span className="text-blue-600 font-bold mt-0.5">•</span>
                <span>Your IP address is registered and monitored for security</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-600 font-bold mt-0.5">•</span>
                <span>Sessions expire after 24 hours of inactivity</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-600 font-bold mt-0.5">•</span>
                <span>Always log out when finished to protect your account</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-600 font-bold mt-0.5">•</span>
                <span>Contact administrator for IP whitelisting requests</span>
              </li>
            </ul>
          </div>
        </main>

        <Footer />
      </div>
    </ProtectedLayout>
  );
}

function FeatureItem({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex items-start gap-3 p-4 bg-background rounded border border-border">
      <div className="text-primary font-bold text-lg mt-0.5">✓</div>
      <div>
        <p className="font-medium text-foreground">{title}</p>
        <p className="text-sm text-secondary">{description}</p>
      </div>
    </div>
  );
}

function getDaysUntilExpiry(expiryDate: string): number {
  const today = new Date();
  const expiry = new Date(expiryDate);
  const diffTime = expiry.getTime() - today.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}
