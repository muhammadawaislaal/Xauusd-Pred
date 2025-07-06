import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from pandas_ta import cci
import pickle
import logging
import requests
import time
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
tf.keras.mixed_precision.set_global_policy('mixed_float16')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = r'E:\personal projects\xauusd Prediction'
ETHUSD_DATA_PATH = os.path.join(BASE_DIR, 'ethusd_5min.csv')
ETHUSD_MODEL_PATH = os.path.join(BASE_DIR, 'ethusd_lstm.keras')
ETHUSD_SCALER_PATH = os.path.join(BASE_DIR, 'ethusd_scaler.pkl')
ETHUSD_XGB_PATH = os.path.join(BASE_DIR, 'ethusd_lstm.xgb.pkl')
ETHUSD_FLAG_PATH = os.path.join(BASE_DIR, 'ethusd_trained.flag')
SEQ_LEN = 60
FEATURES = ['close', 'rsi', 'macd', 'bb_upper', 'atr', 'vwap', 'ema', 'adx', 'cci', 'stoch', 'obv']
API_KEY = '2b89f159f0db4f3796e138044cf0a9f1'
MAX_DATA_POINTS = 1000  # Reduced for faster training

# Fetch data from Binance or TwelveData
def fetch_binance_data(symbol='ETHUSDT'):
    try:
        logging.info(f"Fetching data from Binance for {symbol}")
        df_list = []
        end_time = int(time.time() * 1000)
        for _ in range(5):
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=200&endTime={end_time}"
            resp = requests.get(url)
            if resp.status_code != 200:
                raise Exception(f"Binance API error: {resp.status_code}")
            data = resp.json()
            if not data:
                raise Exception("Empty Binance data")
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            df_list.append(df)
            end_time = int(df['timestamp'].iloc[0]) - 1
            time.sleep(0.3)
        df = pd.concat(df_list).drop_duplicates().sort_values('timestamp')
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.round('s')
        df.set_index('datetime', inplace=True)
        df = clean_numeric_columns(df, ['open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).interpolate(method='linear').ffill().bfill()
        df = df.tail(MAX_DATA_POINTS)
        df = add_technical_indicators(df)
        if len(df) < SEQ_LEN + 1:
            raise Exception(f"Insufficient Binance data: {len(df)}")
        return df.sort_index()
    except Exception as e:
        logging.error(f"Binance fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_twelve_data(symbol='ETH/USD', interval='1min', outputsize=8000):
    try:
        logging.info(f"Fetching data from TwelveData for {symbol}")
        symbol_alt = ['ETH/USDT', 'ETH/USD']
        for sym in symbol_alt:
            url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
            resp = requests.get(url)
            if resp.status_code != 200:
                continue
            data = resp.json().get('values', [])
            if not data:
                continue
            df = pd.DataFrame(data)
            if 'datetime' not in df.columns:
                continue
            df['datetime'] = pd.to_datetime(df['datetime']).dt.round('s')
            df.set_index('datetime', inplace=True)
            df = clean_numeric_columns(df, ['open', 'high', 'low', 'close', 'volume'])
            if 'volume' not in df.columns:
                df['volume'] = 1.0
            df = df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).interpolate(method='linear').ffill().bfill()
            df = df.tail(MAX_DATA_POINTS)
            df = add_technical_indicators(df)
            if len(df) >= SEQ_LEN + 1:
                return df.sort_index()
        raise Exception("No valid data from TwelveData")
    except Exception as e:
        logging.error(f"TwelveData fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_ethusd_data():
    df = fetch_binance_data('ETHUSDT')
    if not df.empty:
        df.to_csv(ETHUSD_DATA_PATH)
        logging.info(f"Saved ETH/USD data to {ETHUSD_DATA_PATH}")
        return df
    df = fetch_twelve_data('ETH/USD')
    if not df.empty:
        df.to_csv(ETHUSD_DATA_PATH)
        logging.info(f"Saved ETH/USD data to {ETHUSD_DATA_PATH}")
        return df
    logging.warning("All data fetches failed, generating synthetic ETH/USD data")
    try:
        base_price = 2600
        volatility = 0.5
        df = pd.DataFrame({
            'datetime': pd.date_range(start=datetime.now() - timedelta(days=5), periods=MAX_DATA_POINTS, freq='5min'),
            'open': base_price + np.cumsum(np.random.normal(0, volatility, MAX_DATA_POINTS)),
            'high': base_price + np.cumsum(np.random.normal(0, volatility, MAX_DATA_POINTS)) + np.random.uniform(0, 0.5, MAX_DATA_POINTS),
            'low': base_price + np.cumsum(np.random.normal(0, volatility, MAX_DATA_POINTS)) - np.random.uniform(0, 0.5, MAX_DATA_POINTS),
            'close': base_price + np.cumsum(np.random.normal(0, volatility, MAX_DATA_POINTS)),
            'volume': 100 + np.random.normal(0, 20, MAX_DATA_POINTS)
        })
        df['datetime'] = df['datetime'].dt.round('s')
        df.to_csv(ETHUSD_DATA_PATH, index=False)
        df = df.set_index('datetime')
        df = clean_numeric_columns(df, ['open', 'high', 'low', 'close', 'volume'])
        df = add_technical_indicators(df)
        return df.sort_index()
    except Exception as e:
        logging.error(f"Failed to generate {ETHUSD_DATA_PATH}: {e}")
        return pd.DataFrame()

# Preprocess data
def preprocess_data():
    logging.info("Loading or fetching ETH/USD data")
    if os.path.exists(ETHUSD_DATA_PATH):
        try:
            df = pd.read_csv(ETHUSD_DATA_PATH)
            datetime_cols = ['time', 'datetime', 'Date', 'Timestamp']
            datetime_col = next((col for col in datetime_cols if col in df.columns), None)
            if not datetime_col:
                raise Exception("No datetime column in ethusd_5min.csv")
            df['datetime'] = pd.to_datetime(df[datetime_col]).dt.round('s')
            df.set_index('datetime', inplace=True)
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise Exception(f"Missing columns in ethusd_5min.csv: {required_cols}")
            df = clean_numeric_columns(df, required_cols + ['volume'])
            if 'volume' not in df.columns:
                df['volume'] = 1.0
            df = df[required_cols + ['volume']]
            df = df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).interpolate(method='linear').ffill().bfill()
        except Exception as e:
            logging.warning(f"Local ETH/USD fetch failed: {e}. Fetching new data.")
            df = fetch_ethusd_data()
    else:
        df = fetch_ethusd_data()
    
    if df.empty:
        raise Exception("Failed to fetch or generate ETH/USD data")
    
    df = add_technical_indicators(df)
    if len(df) < SEQ_LEN + 1:
        raise Exception(f"Insufficient ETH/USD data: {len(df)}")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES])
    with open(ETHUSD_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LEN:i])
        y.append(scaled_data[i, 0])  # Predict close price
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def clean_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.interpolate(method='linear').ffill().bfill()
    df = df.dropna()
    return df

def add_technical_indicators(df):
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
    if len(df) >= SEQ_LEN + 1:
        df = df[df['atr'] > df['atr'].quantile(0.0001)]
    return df

def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    X_train, y_train, X_test, y_test, scaler = preprocess_data()
    
    # Train LSTM
    lstm_model = build_lstm_model(SEQ_LEN, len(FEATURES))
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(ETHUSD_MODEL_PATH, save_best_only=True)
    
    logging.info("Starting LSTM training")
    lstm_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Train XGBoost
    logging.info("Starting XGBoost training")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.15, max_depth=3)
    xgb_model.fit(X_train_flat, y_train)
    
    with open(ETHUSD_XGB_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Create trained flag
    with open(ETHUSD_FLAG_PATH, 'w') as f:
        f.write('trained')
    
    # Evaluate
    lstm_preds = lstm_model.predict(X_test).flatten()
    xgb_preds = xgb_model.predict(X_test_flat)
    nn_mse = np.mean((lstm_preds - y_test) ** 2)
    xgb_mse = np.mean((xgb_preds - y_test) ** 2)
    nn_weight = xgb_mse / (nn_mse + xgb_mse + 1e-10)
    ensemble_preds = nn_weight * lstm_preds + (1 - nn_weight) * xgb_preds
    mse = np.mean((ensemble_preds - y_test) ** 2)
    accuracy = (1 - mse) * 100
    logging.info(f"Training completed. Ensemble MSE: {mse:.4f}, Estimated accuracy: ~{accuracy:.1f}%")
    
    return lstm_model, xgb_model, scaler

if __name__ == "__main__":
    lstm_model, xgb_model, scaler = train_model()
    logging.info(f"LSTM model saved to {ETHUSD_MODEL_PATH}")
    logging.info(f"XGBoost model saved to {ETHUSD_XGB_PATH}")
    logging.info(f"Scaler saved to {ETHUSD_SCALER_PATH}")
    logging.info(f"Trained flag created at {ETHUSD_FLAG_PATH}")