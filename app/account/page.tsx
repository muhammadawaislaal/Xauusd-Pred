'use client'

import { Sidebar } from '@/components/Sidebar'
import { useRouter } from 'next/navigation'
import { CheckCircle, AlertCircle, Zap, Mail, Calendar, TrendingUp } from 'lucide-react'

export default function AccountPage() {
  const router = useRouter()

  const userData = {
    name: 'Trader',
    email: 'trader@example.com',
    avatar: 'A',
  }

  const subscription = {
    plan: 'Golden Plan',
    price: '$100',
    billing: '/month',
    status: 'Active',
    expiryDate: 'August 3, 2024',
    features: [
      'AI Predictions 2× Daily',
      'Real-time Signals (20-min)',
      'XAU/USD & ETH/USD Support',
      'Entry/Stop-Loss/Take-Profit',
      'Advanced Technical Indicators',
      'Priority Support',
    ],
  }

  const usage = {
    predictions: 38,
    remaining: 12,
    total: 50,
  }

  const usagePercentage = (usage.predictions / usage.total) * 100

  return (
    <div className="flex bg-background min-h-screen">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-8 ml-0 md:ml-0">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold text-text-primary mb-2">Account Settings</h1>
            <p className="text-text-muted">Manage your profile and subscription</p>
          </div>

          {/* Profile Section */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <h2 className="text-lg font-semibold text-text-primary mb-6">Profile Information</h2>
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-3xl">{userData.avatar}</span>
              </div>
              <div className="flex-1">
                <h3 className="text-text-primary font-semibold text-lg mb-2">{userData.name}</h3>
                <p className="text-text-muted flex items-center gap-2">
                  <Mail size={16} />
                  {userData.email}
                </p>
              </div>
            </div>
          </div>

          {/* Subscription Section */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-text-primary">Subscription</h2>
              <span className={`px-4 py-2 rounded-full text-sm font-semibold ${subscription.status === 'Active' ? 'bg-signal-buy/20 text-signal-buy flex items-center gap-2' : 'bg-signal-sell/20 text-signal-sell flex items-center gap-2'}`}>
                {subscription.status === 'Active' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                {subscription.status}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Plan Info */}
              <div>
                <p className="text-text-muted text-sm mb-2">Current Plan</p>
                <h3 className="text-text-primary font-semibold text-2xl mb-1">{subscription.plan}</h3>
                <p className="text-text-muted text-lg">
                  {subscription.price}<span className="text-sm">{subscription.billing}</span>
                </p>
              </div>

              {/* Expiry Date */}
              <div className="flex items-center gap-4 p-4 bg-background border border-border rounded-lg">
                <Calendar className="text-accent-primary flex-shrink-0" size={24} />
                <div>
                  <p className="text-text-muted text-sm">Expiry Date</p>
                  <p className="text-text-primary font-semibold">{subscription.expiryDate}</p>
                </div>
              </div>
            </div>

            {/* Features */}
            <div className="mb-6">
              <h4 className="text-text-primary font-semibold mb-4">Plan Features</h4>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {subscription.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-text-muted">
                    <CheckCircle size={18} className="text-signal-buy flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>

            {/* Renew Button */}
            <button className="w-full bg-gradient-to-r from-accent-primary to-accent-secondary hover:shadow-glow-purple text-white font-semibold py-3 rounded-lg transition">
              Renew Subscription
            </button>
          </div>

          {/* Usage Section */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <h2 className="text-lg font-semibold text-text-primary mb-6 flex items-center gap-2">
              <TrendingUp size={20} className="text-accent-primary" />
              API Usage
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-text-primary font-medium">Predictions Used</span>
                  <span className="text-text-muted text-sm">
                    {usage.predictions} / {usage.total}
                  </span>
                </div>
                <div className="w-full bg-background rounded-full h-3 overflow-hidden border border-border">
                  <div
                    className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary transition-all duration-300"
                    style={{ width: `${usagePercentage}%` }}
                  ></div>
                </div>
                <p className="text-text-muted text-xs mt-2">
                  {usage.remaining} predictions remaining until next billing cycle
                </p>
              </div>
            </div>
          </div>

          {/* Settings Section */}
          <div className="bg-surface border border-border rounded-xl p-6">
            <h2 className="text-lg font-semibold text-text-primary mb-6">Preferences</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-background border border-border rounded-lg">
                <div>
                  <p className="text-text-primary font-medium">Email Notifications</p>
                  <p className="text-text-muted text-sm">Receive alerts for new trading signals</p>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    defaultChecked={true}
                    className="w-5 h-5 cursor-pointer"
                  />
                </label>
              </div>

              <div className="flex items-center justify-between p-4 bg-background border border-border rounded-lg">
                <div>
                  <p className="text-text-primary font-medium">Push Notifications</p>
                  <p className="text-text-muted text-sm">Get real-time alerts on your device</p>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    defaultChecked={false}
                    className="w-5 h-5 cursor-pointer"
                  />
                </label>
              </div>

              <div className="flex items-center justify-between p-4 bg-background border border-border rounded-lg">
                <div>
                  <p className="text-text-primary font-medium">Daily Summary</p>
                  <p className="text-text-muted text-sm">Receive a summary of daily predictions</p>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    defaultChecked={true}
                    className="w-5 h-5 cursor-pointer"
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center py-8 text-text-muted text-sm border-t border-border">
            <p>Developed by Muhammad Awais Laal • Educational Project</p>
          </div>
        </div>
      </main>
    </div>
  )
}
