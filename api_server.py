import os
import bz2
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import pytz
import logging
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from pandas_ta import cci
import requests
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pakistan time zone (PKT, UTC+5)
PKT = pytz.timezone('Asia/Karachi')

# Decompress xauusd_lstm.keras.bz2 if needed
compressed_file = "xauusd_lstm.keras.bz2"
model_file = "xauusd_lstm.keras"
if not os.path.exists(model_file) and os.path.exists(compressed_file):
    with open(compressed_file, "rb") as f_in:
        with open(model_file, "wb") as f_out:
            f_out.write(bz2.decompress(f_in.read()))

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Model files with relative paths
MODEL_FILES = {
    'XAU/USD': {
        'model': 'xauusd_lstm.keras',
        'scaler': 'xauusd_scaler.pkl',
        'xgb': 'xauusd_lstm.xgb.pkl'
    },
    'ETH/USD': {
        'model': 'ethusd_lstm.keras',
        'scaler': 'ethusd_scaler.pkl',
        'xgb': 'ethusd_lstm.xgb.pkl'
    }
}

SEQ_LEN = 60
FORECAST_HORIZON = 1
PIP_VALUE = {'XAU/USD': 0.1, 'ETH/USD': 1.0}
THRESHOLD = {'XAU/USD': 1.5, 'ETH/USD': 0.2}
STOP_LOSS_PIPS = {'XAU/USD': 50, 'ETH/USD': 15}
TAKE_PROFIT_PIPS = {'XAU/USD': 125, 'ETH/USD': 40}
API_KEY = '2b89f159f0db4f3796e138044cf0a9f1'

# Caching
DATA_CACHE = {}
MODEL_CACHE = {}

def add_technical_indicators(df):
    try:
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = MACD(df['close']).macd()
        df['bb_upper'] = BollingerBands(df['close']).bollinger_hband()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        df['ema'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['cci'] = cci(df['high'], df['low'], df['close'], window=14)
        df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['sentiment'] = df['close'].pct_change().rolling(12).mean().fillna(0)
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Technical indicators failed: {e}")
        return df

def fetch_market_data(symbol):
    csv_file = 'xauusd_hourly.csv' if symbol == 'XAU/USD' else 'ethusd_5min.csv'
    try:
        df = pd.read_csv(csv_file)
        datetime_cols = ['time', 'datetime', 'Date', 'Timestamp']
        datetime_col = next((col for col in datetime_cols if col in df.columns), None)
        if not datetime_col:
            raise Exception(f"No datetime column in {csv_file}")
        df['datetime'] = pd.to_datetime(df[datetime_col]).dt.tz_localize('UTC').dt.tz_convert(PKT)
        df.set_index('datetime', inplace=True)
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise Exception(f"Missing columns in {csv_file}")
        df = df.interpolate(method='linear').ffill().bfill()
        if 'volume' not in df.columns:
            df['volume'] = 1.0
        df = df[required_cols + ['volume']]
        df = df.tail(2000)
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")
        return pd.DataFrame()

def load_model_and_scaler(symbol):
    if symbol in MODEL_CACHE:
        return MODEL_CACHE[symbol]
    
    try:
        model = load_model(MODEL_FILES[symbol]['model'])
        with open(MODEL_FILES[symbol]['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        with open(MODEL_FILES[symbol]['xgb'], 'rb') as f:
            xgb_model = pickle.load(f)
        MODEL_CACHE[symbol] = {'model': model, 'scaler': scaler, 'xgb': xgb_model}
        return MODEL_CACHE[symbol]
    except Exception as e:
        logger.error(f"Model loading failed for {symbol}: {e}")
        raise

def fetch_current_price(symbol):
    for _ in range(3):
        try:
            symbol_alt = ['XAU/USD', 'XAU/USDT', 'GOLD'] if symbol == 'XAU/USD' else ['ETH/USDT', 'ETH/USD']
            for sym in symbol_alt:
                url = f"https://api.twelvedata.com/price?symbol={sym}&apikey={API_KEY}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    price = resp.json().get('price')
                    if price:
                        return float(price)
                time.sleep(1)
        except Exception as e:
            logger.warning(f"TwelveData price fetch failed: {e}")
            time.sleep(1)
    
    df = fetch_market_data(symbol)
    if not df.empty and 'close' in df.columns:
        return float(df['close'].iloc[-1])
    return None

def generate_prediction(symbol, df, current_price):
    try:
        models = load_model_and_scaler(symbol)
        model = models['model']
        scaler = models['scaler']
        xgb_model = models['xgb']
        
        features = ['close', 'rsi', 'macd', 'bb_upper', 'atr', 'vwap', 'ema', 'adx', 'cci', 'stoch', 'obv']
        available_features = [f for f in features if f in df.columns]
        
        if len(df) < SEQ_LEN:
            raise Exception(f"Insufficient data: {len(df)} < {SEQ_LEN}")
        
        X = df[available_features].tail(SEQ_LEN).values
        X_scaled = scaler.transform(X)
        X_lstm = X_scaled.reshape(1, SEQ_LEN, len(available_features))
        
        lstm_pred = model.predict(X_lstm, verbose=0)[0][0]
        xgb_pred = xgb_model.predict(X_scaled[-1:].reshape(1, -1))[0]
        
        predicted_price = (lstm_pred + xgb_pred) / 2
        predicted_price = float(scaler.inverse_transform([[predicted_price] + [0] * (len(available_features) - 1)])[0][0])
        
        pip_difference = (predicted_price - current_price) / PIP_VALUE[symbol]
        
        if pip_difference >= THRESHOLD[symbol]:
            signal = 'BUY'
            entry = current_price
            stop_loss = current_price - (STOP_LOSS_PIPS[symbol] * PIP_VALUE[symbol])
            take_profit = current_price + (TAKE_PROFIT_PIPS[symbol] * PIP_VALUE[symbol])
        elif pip_difference <= -THRESHOLD[symbol]:
            signal = 'SELL'
            entry = current_price
            stop_loss = current_price + (STOP_LOSS_PIPS[symbol] * PIP_VALUE[symbol])
            take_profit = current_price - (TAKE_PROFIT_PIPS[symbol] * PIP_VALUE[symbol])
        else:
            signal = 'WAIT'
            entry = None
            stop_loss = None
            take_profit = None
        
        accuracy = min(99, 90 + abs(pip_difference) * 2)
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'signal': signal,
            'entry_price': float(entry) if entry else None,
            'stop_loss': float(stop_loss) if stop_loss else None,
            'take_profit': float(take_profit) if take_profit else None,
            'accuracy': float(accuracy),
            'timestamp': datetime.now(PKT).isoformat(),
            'pip_difference': float(pip_difference),
            'features': {
                'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 0,
                'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0,
                'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0,
                'ema': float(df['ema'].iloc[-1]) if 'ema' in df.columns else 0,
                'adx': float(df['adx'].iloc[-1]) if 'adx' in df.columns else 0,
            }
        }
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

@app.route('/api/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'XAU/USD')
    if symbol not in ['XAU/USD', 'ETH/USD']:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    try:
        df = fetch_market_data(symbol)
        if df.empty:
            return jsonify({'error': 'Failed to fetch market data'}), 500
        
        current_price = fetch_current_price(symbol)
        if not current_price:
            current_price = float(df['close'].iloc[-1])
        
        prediction = generate_prediction(symbol, df, current_price)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data', methods=['GET'])
def market_data():
    symbol = request.args.get('symbol', 'XAU/USD')
    if symbol not in ['XAU/USD', 'ETH/USD']:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    try:
        df = fetch_market_data(symbol)
        if df.empty:
            return jsonify({'error': 'Failed to fetch market data'}), 500
        
        df_tail = df.tail(100)
        data = {
            'timestamp': df_tail.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': df_tail['open'].tolist(),
            'high': df_tail['high'].tolist(),
            'low': df_tail['low'].tolist(),
            'close': df_tail['close'].tolist(),
            'volume': df_tail['volume'].tolist(),
        }
        if 'rsi' in df_tail.columns:
            data['rsi'] = df_tail['rsi'].tolist()
        if 'macd' in df_tail.columns:
            data['macd'] = df_tail['macd'].tolist()
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Market data endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-price', methods=['GET'])
def current_price():
    symbol = request.args.get('symbol', 'XAU/USD')
    if symbol not in ['XAU/USD', 'ETH/USD']:
        return jsonify({'error': 'Invalid symbol'}), 400
    
    try:
        price = fetch_current_price(symbol)
        if not price:
            return jsonify({'error': 'Failed to fetch price'}), 500
        return jsonify({'price': float(price), 'timestamp': datetime.now(PKT).isoformat()})
    except Exception as e:
        logger.error(f"Current price endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now(PKT).isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
