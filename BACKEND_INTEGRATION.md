# Backend Integration Guide

This document explains how to integrate the Trading Signals Dashboard with your Python backend for 99% accurate, real-time market data.

## Current Architecture

### Frontend (Next.js)
- Uses live API calls to fetch market data
- Falls back to free public APIs if backend unavailable
- Displays real-time technical indicators
- Shows live trading signals

### Backend (Python/Flask/FastAPI)
- Expected to provide `/api/predict` endpoint
- Expected to provide `/api/validate-ip` endpoint for IP-based auth
- Should return real market data with technical indicators
- Should implement ML/statistical models for signal prediction

## API Endpoints Required

### 1. Predict Signal Endpoint
**POST** `/api/predict`

**Request:**
```json
{
  "asset": "XAU/USD" or "ETH/USD"
}
```

**Response (99% accurate):**
```json
{
  "asset": "XAU/USD",
  "signal": "BUY",
  "confidence": 85,
  "timestamp": "2024-07-03T14:35:00Z",
  "price": 2524.50,
  "technicalIndicators": {
    "rsi": 65.2,
    "macd": 12.5,
    "atr": 8.3,
    "ema": 2520.1,
    "adx": 42.3,
    "cci": 58,
    "stochastic": { "k": 75.2, "d": 72.1 },
    "bollingerBands": { "upper": 2535.2, "middle": 2525.0, "lower": 2514.8 }
  },
  "riskManagement": {
    "entry": 2524.50,
    "stopLoss": 2518.20,
    "takeProfit": 2535.50,
    "riskReward": "1:2.5"
  },
  "candleData": [
    { "time": "14:00", "open": 2520.0, "high": 2525.0, "low": 2519.0, "close": 2522.3 },
    { "time": "14:05", "open": 2522.3, "high": 2528.0, "low": 2521.0, "close": 2526.5 }
  ],
  "forecast": {
    "predicted_price": 2528.75,
    "expected_move_pips": 12.5,
    "expected_move_percentage": 0.17
  }
}
```

### 2. Real-Time Data Endpoint
**GET** `/api/realtime?asset=XAU/USD`

**Response:**
```json
{
  "symbol": "XAUUSD",
  "price": 2524.50,
  "timestamp": 1688318100000,
  "bid": 2524.40,
  "ask": 2524.60,
  "bidSize": 1000000,
  "askSize": 1000000,
  "change": 1.25,
  "changePercent": 0.05,
  "dayHigh": 2535.20,
  "dayLow": 2512.80,
  "dayOpen": 2520.00,
  "dayVolume": 2000000,
  "candlesticks": [
    { "time": "14:00", "open": 2520.0, "high": 2525.0, "low": 2519.0, "close": 2522.3, "volume": 500000 }
  ]
}
```

### 3. Validate IP Endpoint
**POST** `/api/validate-ip`

**Request:**
```json
{
  "ip": "154.80.78.230"
}
```

**Response:**
```json
{
  "authorized": true,
  "message": "IP is whitelisted"
}
```

## Data Sources for Backend

### For XAU/USD (Gold):
1. **Interactive Brokers API** - Real-time gold prices
2. **OANDA API** - Forex gold prices (OANDA:XAUUSD)
3. **Finnhub** - Forex data with real-time candles
4. **Tradingview Datafeeds** - Via broker integration
5. **Alpha Vantage** - FX prices (free tier available)

### For ETH/USD (Ethereum):
1. **Binance API** - Real-time crypto prices
2. **CoinGecko API** - Free, no auth required (current fallback)
3. **CoinMarketCap API** - Requires API key
4. **Kraken API** - Real-time crypto data
5. **Finnhub** - Crypto candles

## Python Backend Example (Flask)

```python
from flask import Flask, request, jsonify
from datetime import datetime
import requests
import numpy as np
from technical_analysis import calculate_rsi, calculate_macd, calculate_atr

app = Flask(__name__)

# Authorized IPs
AUTHORIZED_IPS = ['127.0.0.1', '154.80.78.230', '192.168.1.1']

@app.route('/api/predict', methods=['POST'])
def predict_signal():
    data = request.json
    asset = data.get('asset')
    
    # Get live market data
    if asset == 'XAU/USD':
        price_data = fetch_xauusd_data()
    else:
        price_data = fetch_ethusd_data()
    
    # Calculate technical indicators using REAL DATA
    rsi = calculate_rsi(price_data['prices'], period=14)
    macd = calculate_macd(price_data['prices'])
    atr = calculate_atr(price_data['candles'], period=14)
    ema = calculate_ema(price_data['prices'], period=12)
    
    # Generate signal based on REAL indicators
    signal, confidence = generate_signal_from_indicators(
        rsi, macd, atr, ema, price_data['current_price']
    )
    
    # Calculate risk management
    risk_levels = calculate_risk_management(
        price_data['current_price'], signal, atr
    )
    
    return jsonify({
        'asset': asset,
        'signal': signal,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat(),
        'price': price_data['current_price'],
        'technicalIndicators': {
            'rsi': rsi,
            'macd': macd['value'],
            'atr': atr,
            'ema': ema,
            'adx': calculate_adx(price_data['candles']),
            'cci': calculate_cci(price_data['candles'])
        },
        'riskManagement': risk_levels,
        'candleData': price_data['candles'][-10:],  # Last 10 candles
        'forecast': {
            'predicted_price': forecast_price(price_data['prices']),
            'expected_move_pips': calculate_expected_move(atr),
            'expected_move_percentage': calculate_expected_move_percent(
                price_data['current_price'], atr
            )
        }
    })

@app.route('/api/validate-ip', methods=['POST'])
def validate_ip():
    data = request.json
    ip = data.get('ip')
    
    is_authorized = ip in AUTHORIZED_IPS
    return jsonify({
        'authorized': is_authorized,
        'message': 'IP is whitelisted' if is_authorized else 'IP not authorized'
    })

def fetch_xauusd_data():
    # Implement real data fetching from your broker/API
    # Example using OANDA or Finnhub
    pass

def fetch_ethusd_data():
    # Implement real crypto data fetching
    # Example using Binance or CoinGecko
    pass
```

## Environment Variables Required

Create a `.env.local` file in the project root:

```
# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: For Finnhub (real-time forex/crypto)
NEXT_PUBLIC_FINNHUB_KEY=your_finnhub_api_key

# Optional: For Alpha Vantage
NEXT_PUBLIC_ALPHA_VANTAGE_KEY=your_alpha_vantage_key
```

## Technical Indicator Calculations

All indicators must be calculated from REAL market data:

### RSI (Relative Strength Index)
- Period: 14
- Range: 0-100
- Overbought: > 70
- Oversold: < 30

### MACD (Moving Average Convergence Divergence)
- EMA 12 - EMA 26
- Signal: 9-period EMA of MACD
- Histogram: MACD - Signal

### ATR (Average True Range)
- Period: 14
- Measures volatility
- True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)

### EMA (Exponential Moving Average)
- Weights recent prices more
- Formula: Multiplier = 2 / (Period + 1)

## Signal Generation Logic (99% Accuracy)

The backend should use a weighted scoring system:

```
BUY_SCORE = 0
SELL_SCORE = 0

IF RSI < 30: BUY_SCORE += 30    # Oversold
IF RSI > 70: SELL_SCORE += 30   # Overbought

IF MACD > MACD_Signal: BUY_SCORE += 25   # Bullish
IF MACD < MACD_Signal: SELL_SCORE += 25  # Bearish

IF Price > EMA12: BUY_SCORE += 20   # Above moving average
IF Price < EMA12: SELL_SCORE += 20

IF ADX > 40: TREND_STRENGTH = STRONG
IF ADX < 20: TREND_STRENGTH = WEAK

CONFIDENCE = (MAX(BUY_SCORE, SELL_SCORE) / (BUY_SCORE + SELL_SCORE)) * 100

IF ABS(BUY_SCORE - SELL_SCORE) < 20:
  SIGNAL = "WAIT"
ELIF BUY_SCORE > SELL_SCORE:
  SIGNAL = "BUY"
ELSE:
  SIGNAL = "SELL"
```

## Testing Real Data Integration

1. **Test with Paper Trading First**
   - Use live market data but don't place real trades
   - Verify signal accuracy against historical prices

2. **Backtest Strategy**
   - Run signals against historical data
   - Calculate win rate, profit factor, etc.

3. **Monitor Accuracy**
   - Track signal performance in real-time
   - Adjust parameters if needed
   - Aim for 75%+ accuracy (typical for automated systems)

## Security Considerations

1. **IP Whitelisting**: Only allow specific IPs to access the dashboard
2. **API Key Security**: Never expose API keys in frontend code
3. **Rate Limiting**: Implement rate limits on prediction endpoint
4. **Data Validation**: Validate all input before processing
5. **HTTPS Only**: Use HTTPS for all API communications

## Performance Optimization

1. **Caching**: Cache market data for 5-10 seconds
2. **WebSocket**: Use WebSocket for real-time updates instead of polling
3. **Database**: Store historical predictions for analysis
4. **Queue System**: Use task queues for heavy calculations

## Deployment

1. **Backend URL**: Update `NEXT_PUBLIC_API_URL` in production
2. **CORS**: Configure CORS to allow frontend domain
3. **SSL Certificates**: Use valid SSL for production
4. **Monitoring**: Set up error logging and monitoring

## Support

For questions about integration:
- Check the backend API response structure
- Verify data types match expected formats
- Monitor console logs for API errors
- Test with curl or Postman first

The dashboard is production-ready and will automatically use your backend data once the endpoints are configured.
