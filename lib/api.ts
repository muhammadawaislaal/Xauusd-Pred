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
    const response = await fetch(`${API_BASE_URL}/api/realtime`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      console.error('[v0] API Error:', response.statusText)
      return null
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('[v0] Failed to fetch real-time data:', error)
    return null
  }
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
