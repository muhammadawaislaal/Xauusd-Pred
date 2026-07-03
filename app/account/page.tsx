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
    <div className="flex bg-slate-50 min-h-screen">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-8 ml-0 md:ml-0">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Account Settings</h1>
            <p className="text-slate-600">Manage your profile and subscription</p>
          </div>

          {/* Profile Section */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900 mb-6">Profile Information</h2>
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-3xl">{userData.avatar}</span>
              </div>
              <div className="flex-1">
                <h3 className="text-slate-900 font-semibold text-lg mb-2">{userData.name}</h3>
                <p className="text-slate-600 flex items-center gap-2">
                  <Mail size={16} />
                  {userData.email}
                </p>
              </div>
            </div>
          </div>

          {/* Subscription Section */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-slate-900">Subscription</h2>
              <span className={`px-4 py-2 rounded-full text-sm font-semibold flex items-center gap-2 ${subscription.status === 'Active' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                {subscription.status === 'Active' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                {subscription.status}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Plan Info */}
              <div>
                <p className="text-slate-600 text-sm mb-2">Current Plan</p>
                <h3 className="text-slate-900 font-semibold text-2xl mb-1">{subscription.plan}</h3>
                <p className="text-slate-600 text-lg">
                  {subscription.price}<span className="text-sm">{subscription.billing}</span>
                </p>
              </div>

              {/* Expiry Date */}
              <div className="flex items-center gap-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <Calendar className="text-blue-600 flex-shrink-0" size={24} />
                <div>
                  <p className="text-slate-600 text-sm">Expiry Date</p>
                  <p className="text-slate-900 font-semibold">{subscription.expiryDate}</p>
                </div>
              </div>
            </div>

            {/* Features */}
            <div>
              <p className="text-slate-600 text-sm mb-4 font-medium">INCLUDED FEATURES</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {subscription.features.map((feature, idx) => (
                  <div key={idx} className="flex items-center gap-3 p-3 bg-slate-50 border border-slate-200 rounded-lg">
                    <CheckCircle size={18} className="text-green-600 flex-shrink-0" />
                    <span className="text-slate-900">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Usage Statistics */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                <TrendingUp size={20} />
                API Usage
              </h2>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-600 font-medium">Predictions Used</span>
                  <span className="text-slate-900 font-semibold">{usage.predictions} / {usage.total}</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden border border-slate-300">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                    style={{ width: `${usagePercentage}%` }}
                  ></div>
                </div>
                <p className="text-xs text-slate-500 mt-1">{usage.remaining} predictions remaining</p>
              </div>

              <div className="pt-4 border-t border-slate-300">
                <p className="text-sm text-slate-600 mb-3">Reset on next billing cycle (August 3, 2024)</p>
                <button className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-lg transition">
                  Upgrade Plan
                </button>
              </div>
            </div>
          </div>

          {/* Notifications Settings */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900 mb-6">Notification Preferences</h2>
            <div className="space-y-4">
              <label className="flex items-center gap-3 p-4 border border-slate-300 rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input type="checkbox" defaultChecked className="w-5 h-5 cursor-pointer" />
                <div>
                  <p className="text-slate-900 font-medium">Email Notifications</p>
                  <p className="text-slate-600 text-sm">Receive signal alerts via email</p>
                </div>
              </label>
              <label className="flex items-center gap-3 p-4 border border-slate-300 rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input type="checkbox" defaultChecked className="w-5 h-5 cursor-pointer" />
                <div>
                  <p className="text-slate-900 font-medium">Push Notifications</p>
                  <p className="text-slate-600 text-sm">Real-time trading signal alerts</p>
                </div>
              </label>
              <label className="flex items-center gap-3 p-4 border border-slate-300 rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input type="checkbox" className="w-5 h-5 cursor-pointer" />
                <div>
                  <p className="text-slate-900 font-medium">Daily Summary</p>
                  <p className="text-slate-600 text-sm">Daily trading performance report</p>
                </div>
              </label>
            </div>
          </div>

          {/* Footer */}
          <div className="text-center py-8 text-slate-600 text-sm border-t border-slate-300">
            <p>Developed by Muhammad Awais Laal • Educational Project</p>
          </div>
        </div>
      </main>
    </div>
  )
}
