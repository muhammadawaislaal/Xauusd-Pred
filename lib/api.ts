import axios from 'axios';

// API base URL - will be set to the Python Streamlit backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

export interface PredictionResponse {
  symbol: string;
  current_price: number;
  predicted_price: number;
  signal: 'BUY' | 'SELL' | 'WAIT';
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  accuracy: number;
  timestamp: string;
  pip_difference: number;
  features?: {
    rsi: number;
    macd: number;
    atr: number;
    ema: number;
    adx: number;
  };
}

export interface MarketData {
  timestamp: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  rsi?: number[];
  macd?: number[];
}

// Fetch prediction for a symbol
export async function fetchPrediction(symbol: 'XAU/USD' | 'ETH/USD'): Promise<PredictionResponse> {
  try {
    // Since the backend is Streamlit, we'll use a REST API endpoint
    // For now, we'll generate mock data that matches the backend structure
    // In production, you would create a Flask/FastAPI wrapper
    const response = await axios.get(`/api/predict?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
}

// Fetch historical market data
export async function fetchMarketData(symbol: 'XAU/USD' | 'ETH/USD'): Promise<MarketData> {
  try {
    const response = await axios.get(`/api/market-data?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching market data:', error);
    throw error;
  }
}

// Fetch current price
export async function fetchCurrentPrice(symbol: 'XAU/USD' | 'ETH/USD'): Promise<number> {
  try {
    const response = await axios.get(`/api/current-price?symbol=${symbol}`);
    return response.data.price;
  } catch (error) {
    console.error('Error fetching current price:', error);
    throw error;
  }
}
