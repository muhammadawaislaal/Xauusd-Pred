export interface DashboardData {
  asset: 'XAU/USD' | 'ETH/USD'
  currentPrice: number
  high: number
  low: number
  changePercent: number
  signal: {
    action: 'BUY' | 'SELL' | 'WAIT'
    pips: number
  }
  predictedPrice: number
  confidence: number
  indicators: {
    rsi: { value: number; status: 'Overbought' | 'Neutral' | 'Oversold' }
    macd: { value: number; momentum: 'Bullish' | 'Bearish' | 'Normal' }
    atr: { value: number; volatility: 'High' | 'Normal' | 'Low' }
    ema: { value: number; trend: 'Strong Uptrend' | 'Moderate Trend' | 'Downtrend' }
    adx: { value: number; strength: 'Strong' | 'Moderate' | 'Weak' }
    cci: { value: number; status: 'Overbought' | 'Neutral' | 'Oversold' }
  }
  risk: {
    entry: number
    stopLoss: number
    takeProfit: number
    riskReward: string
  }
  lastSignal: {
    timestamp: string
    currentPrice: number
    action: string
    expectedMove: string
    entry: number
    stopLoss: number
    takeProfit: number
    accuracy: number
  }
  history: Array<{
    date: string
    price: number
    signal: string
    accuracy: number
    status: 'Completed' | 'Pending'
  }>
  candlestickData: Array<{
    time: string
    open: number
    high: number
    low: number
    close: number
  }>
  forecastData: {
    historical: Array<{ time: string; price: number }>
    predicted: Array<{ time: string; price: number }>
  }
}

const mockXAUData: DashboardData = {
  asset: 'XAU/USD',
  currentPrice: 2524.50,
  high: 2535.20,
  low: 2512.80,
  changePercent: 0.45,
  signal: {
    action: 'BUY',
    pips: 2.5,
  },
  predictedPrice: 2528.75,
  confidence: 78,
  indicators: {
    rsi: { value: 65.2, status: 'Neutral' },
    macd: { value: 12.5, momentum: 'Bullish' },
    atr: { value: 8.3, volatility: 'Normal' },
    ema: { value: 2520.1, trend: 'Strong Uptrend' },
    adx: { value: 42.3, strength: 'Strong' },
    cci: { value: 58, status: 'Neutral' },
  },
  risk: {
    entry: 2524.50,
    stopLoss: 2518.20,
    takeProfit: 2535.50,
    riskReward: '1:2.5',
  },
  lastSignal: {
    timestamp: '2024-07-03 14:35:00',
    currentPrice: 2524.50,
    action: 'BUY',
    expectedMove: '+12-15 pips',
    entry: 2524.50,
    stopLoss: 2518.20,
    takeProfit: 2535.50,
    accuracy: 82,
  },
  history: [
    { date: '2024-07-03 14:00', price: 2522.30, signal: 'BUY', accuracy: 85, status: 'Completed' },
    { date: '2024-07-03 13:30', price: 2520.15, signal: 'WAIT', accuracy: 72, status: 'Completed' },
    { date: '2024-07-03 13:00', price: 2519.80, signal: 'SELL', accuracy: 88, status: 'Completed' },
    { date: '2024-07-03 12:30', price: 2521.45, signal: 'BUY', accuracy: 79, status: 'Completed' },
    { date: '2024-07-03 12:00', price: 2519.60, signal: 'WAIT', accuracy: 65, status: 'Completed' },
  ],
  candlestickData: [
    { time: '14:00', open: 2520.0, high: 2525.0, low: 2519.0, close: 2522.3 },
    { time: '14:05', open: 2522.3, high: 2528.0, low: 2521.0, close: 2526.5 },
    { time: '14:10', open: 2526.5, high: 2527.0, low: 2523.0, close: 2524.8 },
    { time: '14:15', open: 2524.8, high: 2528.5, low: 2524.0, close: 2527.2 },
    { time: '14:20', open: 2527.2, high: 2529.0, low: 2525.0, close: 2526.0 },
    { time: '14:25', open: 2526.0, high: 2530.0, low: 2525.5, close: 2529.0 },
    { time: '14:30', open: 2529.0, high: 2531.0, low: 2527.0, close: 2530.5 },
  ],
  forecastData: {
    historical: [
      { time: '13:00', price: 2520.0 },
      { time: '13:15', price: 2519.8 },
      { time: '13:30', price: 2520.5 },
      { time: '13:45', price: 2521.2 },
      { time: '14:00', price: 2522.3 },
      { time: '14:15', price: 2524.8 },
    ],
    predicted: [
      { time: '14:20', price: 2527.0 },
      { time: '14:35', price: 2528.75 },
    ],
  },
}

const mockETHData: DashboardData = {
  asset: 'ETH/USD',
  currentPrice: 3245.80,
  high: 3268.50,
  low: 3210.20,
  changePercent: 1.25,
  signal: {
    action: 'SELL',
    pips: 3.2,
  },
  predictedPrice: 3240.50,
  confidence: 71,
  indicators: {
    rsi: { value: 72.5, status: 'Overbought' },
    macd: { value: -5.2, momentum: 'Bearish' },
    atr: { value: 12.1, volatility: 'High' },
    ema: { value: 3260.5, trend: 'Moderate Trend' },
    adx: { value: 38.7, strength: 'Moderate' },
    cci: { value: 95, status: 'Overbought' },
  },
  risk: {
    entry: 3245.80,
    stopLoss: 3255.20,
    takeProfit: 3220.50,
    riskReward: '1:1.8',
  },
  lastSignal: {
    timestamp: '2024-07-03 14:32:00',
    currentPrice: 3245.80,
    action: 'SELL',
    expectedMove: '-15-18 pips',
    entry: 3245.80,
    stopLoss: 3255.20,
    takeProfit: 3220.50,
    accuracy: 76,
  },
  history: [
    { date: '2024-07-03 14:00', price: 3240.20, signal: 'SELL', accuracy: 80, status: 'Completed' },
    { date: '2024-07-03 13:30', price: 3238.50, signal: 'WAIT', accuracy: 68, status: 'Completed' },
    { date: '2024-07-03 13:00', price: 3235.80, signal: 'BUY', accuracy: 74, status: 'Completed' },
    { date: '2024-07-03 12:30', price: 3242.10, signal: 'SELL', accuracy: 77, status: 'Completed' },
    { date: '2024-07-03 12:00', price: 3238.90, signal: 'BUY', accuracy: 71, status: 'Completed' },
  ],
  candlestickData: [
    { time: '14:00', open: 3240.0, high: 3248.0, low: 3238.0, close: 3240.2 },
    { time: '14:05', open: 3240.2, high: 3250.0, low: 3239.0, close: 3248.5 },
    { time: '14:10', open: 3248.5, high: 3252.0, low: 3245.0, close: 3250.8 },
    { time: '14:15', open: 3250.8, high: 3255.0, low: 3248.0, close: 3252.3 },
    { time: '14:20', open: 3252.3, high: 3260.0, low: 3250.0, close: 3258.5 },
    { time: '14:25', open: 3258.5, high: 3268.5, low: 3257.0, close: 3265.2 },
    { time: '14:30', open: 3265.2, high: 3268.0, low: 3260.0, close: 3263.0 },
  ],
  forecastData: {
    historical: [
      { time: '13:00', price: 3235.0 },
      { time: '13:15', price: 3237.5 },
      { time: '13:30', price: 3238.5 },
      { time: '13:45', price: 3242.0 },
      { time: '14:00', price: 3240.2 },
      { time: '14:15', price: 3250.8 },
    ],
    predicted: [
      { time: '14:20', price: 3245.0 },
      { time: '14:35', price: 3240.50 },
    ],
  },
}

export function getMockData(asset: 'XAU/USD' | 'ETH/USD'): DashboardData {
  return asset === 'XAU/USD' ? mockXAUData : mockETHData
}
