'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Eye, EyeOff, Loader2 } from 'lucide-react'

export default function LoginPage() {
  const router = useRouter()
  const [password, setPassword] = useState('')
  const [ipAddress, setIpAddress] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    // Fetch user's IP address
    const fetchIP = async () => {
      try {
        const response = await fetch('https://api.ipify.org?format=json')
        const data = await response.json()
        setIpAddress(data.ip)
      } catch (err) {
        setIpAddress('Unable to detect')
      }
    }
    fetchIP()
  }, [])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    // Simulate login validation
    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      setIsLoading(false)
      return
    }

    // Mock authentication - replace with actual backend call
    if (password === 'demo123') {
      setTimeout(() => {
        router.push('/dashboard')
      }, 500)
    } else {
      setError('Invalid password')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-[#1a1a2e] flex items-center justify-center px-4">
      {/* Background gradient elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-accent-primary/20 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-accent-secondary/20 to-transparent rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 w-full max-w-md">
        <div className="bg-surface border border-border rounded-xl shadow-2xl p-8">
          {/* Header */}
          <div className="mb-8 text-center">
            <div className="inline-block mb-4 p-3 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-lg">
              <span className="text-white font-bold text-2xl">⚡</span>
            </div>
            <h1 className="text-2xl font-bold text-text-primary mb-2">Trading Signals</h1>
            <p className="text-text-muted text-sm">AI-Powered XAU/USD & ETH/USD Predictions</p>
          </div>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-4">
            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-text-primary mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted/50 focus:outline-none focus:ring-2 focus:ring-accent-primary focus:border-transparent transition"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary transition"
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            {/* IP Address */}
            <div>
              <label htmlFor="ip" className="block text-sm font-medium text-text-primary mb-2">
                IP Address
              </label>
              <input
                id="ip"
                type="text"
                value={ipAddress}
                readOnly
                className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-muted cursor-not-allowed opacity-60 focus:outline-none"
              />
              <p className="text-xs text-text-muted mt-1">Automatically detected for subscription verification</p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-signal-sell/10 border border-signal-sell/30 rounded-lg p-3 text-signal-sell text-sm">
                {error}
              </div>
            )}

            {/* Sign In Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-accent-primary to-accent-secondary hover:shadow-glow-purple disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Demo info */}
          <div className="mt-6 pt-6 border-t border-border">
            <p className="text-xs text-text-muted text-center">
              Demo password: <code className="bg-background px-2 py-1 rounded text-text-primary font-mono">demo123</code>
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-text-muted text-xs mt-6">
          Developed by Muhammad Awais Laal • Educational Project
        </p>
      </div>
    </div>
  )
}
