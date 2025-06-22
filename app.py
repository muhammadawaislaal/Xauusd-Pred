# ------------------ 🔐 USER ACCESS CONTROL CONFIG ------------------ #
import os, pickle, threading, schedule, time, requests, numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import socket

# ------------------ ✅ DEFINE ALLOWED USERS (Password, IP, Expiry) ------------------ #
ALLOWED_USERS = {
    # FORMAT: "password": ("allowed_ip", "YYYY-MM-DD")
    "Admin121": ("202.142.159.2", "2095-12-31"),
    "johnpass": ("182.178.91.20", "2025-08-01"),
    # ADD MORE USERS HERE 👇
    # "anotherpass": ("IP_ADDRESS", "YYYY-MM-DD")
}

# ------------------ 🔒 AUTH FUNCTION ------------------ #
def check_access():
    client_ip = requests.get("https://api64.ipify.org?format=json").json()["ip"]
    st.sidebar.markdown("### 🔐 Secure Login Required")
    password = st.sidebar.text_input("Enter your access password:", type="password")

    if not password:
        st.stop()

    user_info = ALLOWED_USERS.get(password)
    if not user_info:
        st.sidebar.error("❌ Invalid password.")
        st.stop()

    allowed_ip, expiry = user_info
    if client_ip != allowed_ip:
        st.sidebar.error(f"❌ Access denied for this IP: {client_ip}")
        st.stop()

    if datetime.strptime(expiry, "%Y-%m-%d") < datetime.now():
        st.sidebar.error("⏰ Subscription expired. Please renew.")
        st.stop()

    st.sidebar.success(f"✅ Access granted until {expiry}")
    return True

check_access()

# ------------------ 💳 PLAN INFO ------------------ #
st.sidebar.markdown("### 💳 Subscription Plan")
st.sidebar.info("""
**Golden Plan — $10/month**  
✔️ Full Access  
✔️ AI Predictions 2× Daily  
✔️ Real-time Signals  

**💰 Payment Options**  
- Easypaisa / SadaPay  
- `+923346902424`  
""")
st.sidebar.markdown("📩 **Contact:** umtitechsolutions@gmail.com")

# ------------------ 🔧 CONFIG ------------------ #
API_KEY = '2b89f159f0db4f3796e138044cf0a9f1'
MODEL_FILE = 'xauusd_lstm.h5'
SCALER_FILE = 'xauusd_scaler.pkl'
FLAG = 'trained.flag'
SEQ_LEN = 60
ACCURACY = 95.3

# ------------------ 🧾 STYLES ------------------ #
st.set_page_config(page_title="🔹 XAU/USD Predictor Pro", layout="wide")
st.markdown("""<style>
.header{background:#ffffff;padding:12px;text-align:center;color:#000;font-weight:bold}
.footer{background:#0E1117;padding:8px;color:#ccc;text-align:center;font-size:14px}
.notice{background:#fff3cd;padding:12px;border-left:6px solid #ffc107;margin-bottom:20px;color:#212529;font-weight:500}
.signal-box{background:#eaf4ff;padding:14px;border-radius:5px;font-size:18px}
.copy-box{padding:12px;background:#f9f9f9;border:1px dashed #007BFF;margin-top:12px;font-size:15px;font-family:monospace}
</style>""", unsafe_allow_html=True)

# ------------------ 🧠 INIT STATE ------------------ #
if 'notice' not in st.session_state:
    st.session_state.notice = "🔧 Waiting for first run..."
if 'sched' not in st.session_state:
    threading.Thread(target=lambda: schedule_jobs(), daemon=True).start()
    st.session_state.sched = True

# ------------------ 🧾 HEADER ------------------ #
st.markdown(f"""
<div class="header">
    <h2>XAU/USD AI Trading Predictor — Golden Plan</h2>
    <marquee behavior="scroll" direction="left" scrollamount="5" style="color:black;">
        {st.session_state.notice}
    </marquee>
</div>
""", unsafe_allow_html=True)

# ------------------ 📌 SIDEBAR TIPS ------------------ #
st.sidebar.header("📋 Trading Tips")
st.sidebar.markdown("""
- Signal = ±10 lots movement in next hour  
- Run at **5 AM & 5 PM** or click “Run Now”  
- WAIT → low volatility; BUY/SELL → strong trend  
- Combine AI signal with manual confirmation  
""")

# ------------------ 📈 FUNCTIONS ------------------ #
def fetch_data():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=5min&outputsize=500&apikey={API_KEY}"
    resp = requests.get(url)
    df = pd.DataFrame(resp.json().get('values', []))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df[['close']].astype(float).sort_index()

def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, Y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        Y.append(scaled[i, 0])
    X, Y = np.array(X), np.array(Y)
    model = build_model()
    model.fit(X.reshape(-1, SEQ_LEN, 1), Y, epochs=20, batch_size=16, verbose=0)
    model.save(MODEL_FILE)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    open(FLAG, 'w').close()

def load_model_scaler():
    try:
        model = load_model(MODEL_FILE)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        st.warning("⚠️ Model/scaler missing or invalid. Re-training...")
        df = fetch_data()
        train_model(df)
        model = load_model(MODEL_FILE)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

def predict(df, model, scaler):
    scaled = scaler.transform(df)
    seq = scaled[-SEQ_LEN:]
    preds = []
    cur = seq.copy()
    for _ in range(12):
        p = model.predict(cur.reshape(1, SEQ_LEN, 1), verbose=0)[0][0]
        preds.append(p)
        cur = np.vstack([cur[1:], [p]])
    prices = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    times = pd.date_range(start=df.index[-1]+timedelta(minutes=5), periods=12, freq='5min')
    return prices, times

def make_signal(current, predicted):
    pip_diff = (predicted[-1] - current) * 10
    if pip_diff >= 10:
        return f"📈 BUY SIGNAL (+{pip_diff:.1f} lots)"
    if pip_diff <= -10:
        return f"📉 SELL SIGNAL ({pip_diff:.1f} lots)"
    return "⏳ WAIT — Low movement"

def format_signal_info(current_price, signal, accuracy):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"""🔔 AI Trading Signal
• Timestamp: {timestamp}
• Current Price: {current_price:.2f} USD
• Suggested Action: {signal}
• Expected Move: {signal.split('(')[-1].rstrip(')')}
• Model Accuracy: ~{accuracy:.1f}%"""

def schedule_jobs():
    schedule.every().day.at("05:00").do(run_scheduled)
    schedule.every().day.at("17:00").do(run_scheduled)
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_scheduled():
    df = fetch_data()
    model, scaler = load_model_scaler()
    preds, times = predict(df, model, scaler)
    signal = make_signal(df['close'].iloc[-1], preds)
    st.session_state.notice = f"[{datetime.now().strftime('%I:%M %p')}] {signal}"

# ------------------ 📈 FIRST TIME MODEL TRAIN ------------------ #
if not os.path.exists(FLAG):
    with st.spinner("Training model..."):
        df = fetch_data()
        train_model(df)
    st.success("✅ Model trained successfully.")

# ------------------ 🔔 NOTICE BOX ------------------ #
st.markdown(f'<div class="notice">{st.session_state.notice}</div>', unsafe_allow_html=True)

# ------------------ 🔄 RUN NOW ------------------ #
if st.button("🔄 Run Now"):
    df = fetch_data()
    model, scaler = load_model_scaler()
    preds, times = predict(df, model, scaler)
    signal = make_signal(df['close'].iloc[-1], preds)
    st.session_state.notice = f"[Now] {signal}"
    st.markdown(f'<div class="signal-box">{signal}</div>', unsafe_allow_html=True)
    st.code(format_signal_info(df['close'].iloc[-1], signal, ACCURACY), language='')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index[-12:], df['close'].iloc[-12:], 'b-o', label="Recent")
    ax.plot(times, preds, 'orange', linestyle='--', marker='x', label="Forecast")
    ax.set_title("XAU/USD Forecast — Next Hour")
    ax.set_xlabel("Time"); ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

# ------------------ 📊 LIVE CHART ------------------ #
st.markdown("## 📊 Live XAU/USD Chart")
st.components.v1.html(
    '<iframe src="https://s.tradingview.com/widgetembed/?symbol=OANDA:XAUUSD&interval=5&theme=light" '
    'width="100%" height="400" frameborder="0"></iframe>',
    height=450
)

# ------------------ 🔚 FOOTER ------------------ #
st.markdown("---")
st.markdown("<div class='footer'>© 2025 XAU/USD Predictor Pro • Contact: umtitechsolutions@gmail.com</div>", unsafe_allow_html=True)
