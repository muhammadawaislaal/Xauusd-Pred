'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { ProtectedLayout } from '@/components/ProtectedLayout';
import Sidebar from '@/components/Sidebar';
import DashboardHeader from '@/components/DashboardHeader';
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
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-secondary">Loading account...</p>
        </div>
      </div>
    );
  }

  return (
    <ProtectedLayout>
      <div className="flex min-h-screen bg-background">
        <Sidebar />
        
        <main className="flex-1 lg:ml-0">
          <div className="p-4 sm:p-6 lg:p-8 max-w-4xl mx-auto">
            {/* Page Header */}
            <DashboardHeader title="Account Settings" />

            {/* Account Info Card */}
            <div className="bg-card-bg border border-border rounded-xl p-8 mb-8 card-shadow">
              <h2 className="text-2xl font-bold text-foreground mb-8">Account Information</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Left Column */}
                <div className="space-y-6">
                  {/* Username */}
                  <div>
                    <label className="block text-sm font-semibold text-secondary mb-3">Username</label>
                    <div className="px-4 py-3 bg-background rounded-lg border border-border">
                      <p className="font-mono text-foreground font-semibold">{user?.username}</p>
                    </div>
                  </div>

                  {/* IP Address */}
                  <div>
                    <label className="block text-sm font-semibold text-secondary mb-3">Registered IP Address</label>
                    <div className="px-4 py-3 bg-background rounded-lg border border-border">
                      <p className="font-mono text-foreground font-semibold break-all text-sm">{user?.ip}</p>
                    </div>
                  </div>
                </div>

                {/* Right Column - Subscription Info */}
                <div>
                  <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-lg p-6">
                    <h3 className="text-lg font-bold text-foreground mb-6">Subscription Status</h3>
                    
                    {user?.subscription ? (
                      <div className="space-y-4">
                        {/* Status */}
                        <div>
                          <p className="text-xs font-semibold text-secondary mb-2">Status</p>
                          <div className="flex items-center gap-2">
                            <div className={`h-3 w-3 rounded-full ${user.subscription.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                            <span className="font-semibold text-foreground capitalize">{user.subscription.status}</span>
                          </div>
                        </div>

                        {/* Plan */}
                        {user.subscription.plan && (
                          <div>
                            <p className="text-xs font-semibold text-secondary mb-2">Plan</p>
                            <p className="font-semibold text-foreground capitalize bg-gradient-to-r from-purple-500/20 to-blue-500/20 px-3 py-2 rounded-lg inline-block">{user.subscription.plan}</p>
                          </div>
                        )}

                        {/* Expiry Date */}
                        {user.subscription.expiryDate && (
                          <div>
                            <p className="text-xs font-semibold text-secondary mb-2">Expiry Date</p>
                            <p className="font-semibold text-foreground">
                              {new Date(user.subscription.expiryDate).toLocaleDateString('en-US', {
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric',
                              })}
                            </p>
                            {getDaysUntilExpiry(user.subscription.expiryDate) <= 30 && (
                              <p className="text-xs text-amber-400 mt-3 font-medium">
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
            <div className="bg-card-bg border border-border rounded-xl p-8 mb-8 card-shadow">
              <h2 className="text-2xl font-bold text-foreground mb-8">Active Features</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FeatureItem title="Real-time Predictions" description="Live AI-powered price predictions" />
                <FeatureItem title="Trading Signals" description="BUY/SELL/HOLD signals with confidence levels" />
                <FeatureItem title="Technical Analysis" description="5 technical indicators (RSI, MACD, ATR, EMA, ADX)" />
                <FeatureItem title="Risk Management" description="Stop Loss and Take Profit calculations" />
                <FeatureItem title="Multi-Asset Support" description="Trade XAU/USD and ETH/USD" />
                <FeatureItem title="Auto-Refresh" description="5-minute automatic data updates" />
              </div>
            </div>

            {/* Security Info */}
            <div className="bg-gradient-to-br from-blue-500/10 to-blue-500/5 border border-blue-500/20 rounded-xl p-8 mb-8 card-shadow">
              <h2 className="text-xl font-bold text-blue-400 mb-6 flex items-center gap-2">
                <span>🔐</span>
                Security Notice
              </h2>
              <ul className="space-y-3 text-sm text-secondary">
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold mt-0.5 flex-shrink-0">✓</span>
                  <span>Your IP address is registered and monitored for security</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold mt-0.5 flex-shrink-0">✓</span>
                  <span>Sessions expire after 24 hours of inactivity</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold mt-0.5 flex-shrink-0">✓</span>
                  <span>Always log out when finished to protect your account</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold mt-0.5 flex-shrink-0">✓</span>
                  <span>Contact administrator for IP whitelisting requests</span>
                </li>
              </ul>
            </div>

            {/* Footer */}
            <Footer />
          </div>
        </main>
      </div>
    </ProtectedLayout>
  );
}

function FeatureItem({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex items-start gap-4 p-5 bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-lg hover:border-purple-500/40 transition-all">
      <div className="text-purple-400 font-bold text-xl mt-0.5 flex-shrink-0">✓</div>
      <div>
        <p className="font-semibold text-foreground">{title}</p>
        <p className="text-sm text-secondary mt-1">{description}</p>
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
