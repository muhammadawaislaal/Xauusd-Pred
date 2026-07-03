# Backend API Specification

## Overview
The trading dashboard requires a backend API server to provide real-time market analysis, AI predictions, and technical indicators. All data must be 99% accurate and synchronized with live TradingView market data.

## Required Endpoints

### 1. POST /api/predict
**Purpose**: Generate AI trading signal for a specific asset

**Request Body**:
```json
{
  "asset": "XAU/USD" | "ETH/USD"
}
```

**Response** (200 OK):
```json
{
  "asset": "XAU/USD",
  "signal": "BUY" | "SELL" | "WAIT",
  "confidence": 0.95,
  "timestamp": "2024-01-15T14:23:45Z",
  "price": 2524.50,
  "high": 2535.20,
  "low": 2518.30,
  "changePercent": 0.45,
  "technicalIndicators": {
    "rsi": {
      "value": 65.4,
      "status": "Neutral" | "Overbought" | "Oversold"
    },
    "macd": {
      "value": 12.3,
      "momentum": "Bullish" | "Bearish" | "Normal"
    },
    "atr": {
      "value": 8.5,
      "volatility": "High" | "Normal" | "Low"
    },
    "ema": {
      "value": 2520.1,
      "trend": "Strong Uptrend" | "Moderate Trend" | "Downtrend"
    },
    "adx": {
      "value": 32.1,
      "strength": "Strong" | "Moderate" | "Weak"
    },
    "cci": {
      "value": 145.2,
      "status": "Overbought" | "Neutral" | "Oversold"
    }
  },
  "riskManagement": {
    "entry": 2524.50,
    "stopLoss": 2518.20,
    "takeProfit": 2535.50,
    "riskReward": "1:2.5",
    "pips": 2.5
  },
  "candleData": [
    {
      "time": "14:20",
      "open": 2522.30,
      "high": 2525.80,
      "low": 2521.10,
      "close": 2524.50
    }
  ],
  "forecastData": [
    {
      "time": "14:35",
      "predicted": 2528.75,
      "confidence": 0.92
    }
  ],
  "analysisReport": "Technical analysis shows...",
  "accuracy": 0.99
}
```

### 2. GET /api/market/{asset}
**Purpose**: Get current live market data synchronized with TradingView

**Request Parameters**:
- `asset`: XAU_USD or ETH_USD

**Response** (200 OK):
```json
{
  "asset": "XAU/USD",
  "price": 2524.50,
  "high": 2535.20,
  "low": 2518.30,
  "changePercent": 0.45,
  "timestamp": "2024-01-15T14:23:45Z",
  "bid": 2524.40,
  "ask": 2524.60,
  "volume": 1250000
}
```

### 3. GET /api/history/{asset}?limit=20
**Purpose**: Get historical prediction accuracy and performance metrics

**Response** (200 OK):
```json
[
  {
    "timestamp": "2024-01-15T14:20:00Z",
    "signal": "BUY",
    "confidence": 0.95,
    "entryPrice": 2524.50,
    "exitPrice": 2528.75,
    "pips": 4.25,
    "accuracy": 0.99,
    "status": "Completed" | "Pending"
  }
]
```

### 4. GET /api/health
**Purpose**: Health check endpoint

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:23:45Z"
}
```

## Authentication
All requests (except /api/health) require Bearer token in Authorization header:
```
Authorization: Bearer {auth_token}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid asset parameter"
}
```

### 401 Unauthorized
```json
{
  "error": "Invalid or expired authentication token"
}
```

### 503 Service Unavailable
```json
{
  "error": "Backend service temporarily unavailable"
}
```

## Configuration

Set environment variables on the frontend:
```bash
NEXT_PUBLIC_API_URL=http://your-backend-server:5000
NEXT_PUBLIC_WS_URL=ws://your-backend-server:5000/ws
```

## Data Quality Requirements

1. **Price Accuracy**: Must be synchronized with live TradingView data (±0.01% tolerance)
2. **Technical Indicators**: Calculate using standard formulas:
   - RSI: 14-period relative strength index
   - MACD: 12/26/9 exponential moving average convergence divergence
   - ATR: 14-period average true range
   - EMA: 20/50-period exponential moving average
   - ADX: 14-period average directional index
   - CCI: 20-period commodity channel index

3. **Signal Accuracy**: ≥99% for professional trading use
4. **Latency**: < 2 seconds for prediction endpoint response

## Response Timeouts
- Prediction: 15 seconds
- Market data: 8 seconds
- History: 10 seconds
- Health check: 5 seconds

## Testing

Use the included test configuration to verify backend connectivity:
1. Ensure backend is running on configured URL
2. Check `/api/health` endpoint
3. Authenticate with valid token
4. Call `/api/predict` with test asset

## Notes
- This frontend contains NO mock data fallbacks
- Backend API is REQUIRED for all functionality
- All displayed data must be from live market sources
- Intended for professional trading use with real capital
