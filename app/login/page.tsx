'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Eye, EyeOff, Loader2, AlertCircle, CheckCircle } from 'lucide-react'
import { validateIP } from '@/lib/api'

export default function LoginPage() {
  const router = useRouter()
  const [password, setPassword] = useState('')
  const [ipAddress, setIpAddress] = useState('Detecting...')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [ipAuthorized, setIpAuthorized] = useState<boolean | null>(null)

  useEffect(() => {
    // Fetch user's IP address and validate it
    const fetchAndValidateIP = async () => {
      try {
        const response = await fetch('https://api.ipify.org?format=json')
        const data = await response.json()
        setIpAddress(data.ip)
        
        // Try backend API first, then fall back to local allowlist
        let isAuthorized = await validateIP(data.ip)
        
        // If backend API fails, use local allowlist as fallback
        if (!isAuthorized) {
          const LOCAL_AUTHORIZED_IPS = ['127.0.0.1', '::1', '192.168.1.1', '203.0.113.45', '18.219.13.193', '::ffff:127.0.0.1', '154.80.78.230']
          isAuthorized = LOCAL_AUTHORIZED_IPS.includes(data.ip)
          if (isAuthorized) {
            console.log('[v0] IP authorized via local allowlist:', data.ip)
          }
        }
        
        setIpAuthorized(isAuthorized)
        console.log('[v0] IP validation result:', { ip: data.ip, authorized: isAuthorized })
      } catch (err) {
        console.error('[v0] Error detecting IP:', err)
        setIpAddress('Unable to detect')
        setIpAuthorized(false)
      }
    }
    fetchAndValidateIP()
  }, [])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      // Check if IP is authorized
      if (!ipAuthorized) {
        setError('Your IP address is not authorized for this subscription. Contact support to whitelist your IP.')
        setIsLoading(false)
        return
      }

      // Validate password
      if (password.length < 6) {
        setError('Password must be at least 6 characters')
        setIsLoading(false)
        return
      }

      // Authenticate with backend API (optional - can be added later)
      // const loginResponse = await fetch(`${API_BASE_URL}/api/login`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ password, ip: ipAddress }),
      // })

      // For now, validate password locally
      if (password === 'Admin121') {
        // Store auth token in localStorage
        localStorage.setItem('auth_token', 'demo_token_' + Date.now())
        localStorage.setItem('user_ip', ipAddress)
        
        console.log('[v0] Login successful for IP:', ipAddress)
        
        setTimeout(() => {
          router.push('/dashboard')
        }, 500)
      } else {
        setError('Invalid password. Please try again.')
        setIsLoading(false)
      }
    } catch (err) {
      console.error('[v0] Login error:', err)
      setError('An error occurred during login. Please try again.')
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center px-4">
      {/* Subtle background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-100/30 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-100/20 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 w-full max-w-md">
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-8">
          {/* Header */}
          <div className="mb-8 text-center">
            <div className="inline-block mb-4 p-3 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
              <span className="text-white font-bold text-2xl">📊</span>
            </div>
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Trading Signals</h1>
            <p className="text-slate-600 text-sm">AI-Powered XAU/USD & ETH/USD Predictions</p>
          </div>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-5">
            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-slate-700 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-3 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700 transition"
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            {/* IP Address */}
            <div>
              <label htmlFor="ip" className="block text-sm font-semibold text-slate-700 mb-2">
                IP Address
              </label>
              <div className="relative">
                <input
                  id="ip"
                  type="text"
                  value={ipAddress}
                  readOnly
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg px-4 py-3 text-slate-600 cursor-not-allowed focus:outline-none"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  {ipAuthorized === true && <CheckCircle size={20} className="text-green-500" />}
                  {ipAuthorized === false && <AlertCircle size={20} className="text-red-500" />}
                </div>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                {ipAuthorized === true ? '✓ IP is authorized' : ipAuthorized === false ? '✗ IP not authorized' : 'Checking authorization...'}
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm flex gap-2">
                <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Sign In Button */}
            <button
              type="submit"
              disabled={isLoading || ipAuthorized === false}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
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

          {/* Info footer */}
          <div className="mt-6 pt-6 border-t border-slate-200">
            <p className="text-xs text-slate-400 text-center">
              IP-based authentication enabled
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-slate-600 text-xs mt-6">
          Developed by Muhammad Awais Laal • Educational Project
        </p>
      </div>
    </div>
  )
}
