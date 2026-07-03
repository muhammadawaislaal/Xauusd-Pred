// API service for communicating with the backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface SignalResponse {
  asset: string
  signal: 'BUY' | 'SELL' | 'WAIT'
  confidence: number
  timestamp: string
  price: number
  technicalIndicators: {
    rsi: number
    macd: number
    atr: number
    ema: number
    adx: number
    cci: number
  }
  riskManagement: {
    entry: number
    stopLoss: number
    takeProfit: number
    riskReward: string
  }
}

export async function getPredictedSignal(asset: 'XAU/USD' | 'ETH/USD'): Promise<SignalResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        asset: asset,
      }),
    })

    if (!response.ok) {
      console.error('[v0] API Error:', response.statusText)
      return null
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('[v0] Failed to fetch predicted signal:', error)
    return null
  }
}

export async function getRealTimeData(asset: 'XAU/USD' | 'ETH/USD'): Promise<any | null> {
  try {
    // First try backend API
    const response = await fetch(`${API_BASE_URL}/api/realtime`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (response.ok) {
      const data = await response.json()
      console.log('[v0] Real-time data from backend API:', asset)
      return data
    }

    // Fallback to live market APIs
    if (asset === 'XAU/USD') {
      return await getXAUUSDRealTime()
    } else {
      return await getETHUSDRealTime()
    }
  } catch (error) {
    console.error('[v0] Failed to fetch real-time data:', error)
    return null
  }
}

// Fetch XAU/USD live data from market APIs
async function getXAUUSDRealTime(): Promise<any | null> {
  try {
    // Try metals.live API (free, no auth required)
    const response = await fetch('https://api.metals.live/v1/spot/gold')
    if (response.ok) {
      const data = await response.json()
      console.log('[v0] Live XAU/USD data from metals.live API')
      return {
        symbol: 'XAUUSD',
        price: data.price?.usd || 2500,
        timestamp: Date.now(),
        change: data.change?.usd || 0,
        changePercent: (data.change?.usd / (data.price?.usd || 1)) * 100 || 0,
      }
    }
  } catch (error) {
    console.error('[v0] Error fetching XAU/USD data:', error)
  }
  return null
}

// Fetch ETH/USD live data from CoinGecko
async function getETHUSDRealTime(): Promise<any | null> {
  try {
    const response = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&include_last_updated_at=true'
    )
    if (response.ok) {
      const data = await response.json()
      console.log('[v0] Live ETH/USD data from CoinGecko API')
      return {
        symbol: 'ETHUSDT',
        price: data.ethereum?.usd || 3200,
        timestamp: data.ethereum?.last_updated_at * 1000 || Date.now(),
        change: (data.ethereum?.usd * data.ethereum?.usd_24h_change) / 100 || 0,
        changePercent: data.ethereum?.usd_24h_change || 0,
        marketCap: data.ethereum?.usd_market_cap || 0,
        volume24h: data.ethereum?.usd_24h_vol || 0,
      }
    }
  } catch (error) {
    console.error('[v0] Error fetching ETH/USD data:', error)
  }
  return null
}

export async function validateIP(ip: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/validate-ip`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ip: ip,
      }),
    })

    if (!response.ok) {
      return false
    }

    const data = await response.json()
    return data.authorized || false
  } catch (error) {
    console.error('[v0] Failed to validate IP:', error)
    return false
  }
}
