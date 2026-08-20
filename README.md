
</div>

## 📋 Table of Contents
- [✨ Features](#features)
- [🚀 Quick Start](#quick-start)
- [📁 Project Structure](#project-structure)
- [🛠️ Installation](#installation)
- [⚙️ Configuration](#configuration)
- [📊 Usage Guide](#usage-guide)
- [🧠 AI Models](#ai-models)
- [📈 Technical Indicators](#technical-indicators)
- [💰 Trading Signals](#trading-signals)
- [🔒 Security](#security)
- [🚀 Deployment](#deployment)
- [🤝 Contributing](#contributing)
- [📄 License](#license)
- [⚠️ Disclaimer](#disclaimer)
- [👨‍💻 Developer](#developer)

## ✨ Features

### 🎯 Core Features
- **🤖 Dual AI Model Prediction** - LSTM + XGBoost ensemble for enhanced accuracy
- **📊 Multi-Asset Support** - XAU/USD (Gold) & ETH/USD (Ethereum) predictions
- **⚡ Real-time Data Fetching** - Multiple data sources with fallback mechanisms
- **📈 Automated Technical Analysis** - 10+ technical indicators integrated
- **🚦 Smart Trading Signals** - BUY/SELL/WAIT recommendations with risk management

### 🛡️ Professional Features
- **🎯 Precision Timing** - 20-minute ahead price predictions
- **📊 Interactive Charts** - Plotly candlestick charts with TradingView integration
- **🔔 Signal Alerts** - Entry, Stop-Loss, Take-Profit levels
- **📱 Responsive Dashboard** - Streamlit-based professional interface
- **🔄 Automatic Scheduling** - 5 AM & 5 PM PKT predictions

### 🚀 Technical Highlights
- Hybrid LSTM + XGBoost model architecture
- Multi-source data aggregation (Binance, TwelveData, local)
- Advanced technical indicator calculation
- Real-time market sentiment analysis
- Pakistan timezone (PKT) optimized scheduling

## 📁 Project Structure

```
Xauusd-Pred/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── models/                             # AI models directory
│   ├── xauusd_lstm.keras              # XAU/USD LSTM model (compressed)
│   ├── xauusd_lstm.xgb.pkl            # XAU/USD XGBoost model
│   ├── xauusd_scaler.pkl              # XAU/USD data scaler
│   ├── ethusd_lstm.keras              # ETH/USD LSTM model
│   ├── ethusd_lstm.xgb.pkl            # ETH/USD XGBoost model
│   └── ethusd_scaler.pkl              # ETH/USD data scaler
│
├── data/                               # Market data
│   ├── xauusd_hourly.csv              # Historical XAU/USD data
│   └── ethusd_5min.csv                # Historical ETH/USD data
│
├── training/                           # Model training scripts
│   ├── train_lstm.py                  # LSTM model training
│   ├── train_xgb.py                   # XGBoost model training
│   └── data_preprocessing.py          # Data preprocessing utilities
│
├── utils/                              # Utility functions
│   ├── data_fetcher.py               # Multi-source data fetching
│   ├── technical_indicators.py       # Technical analysis calculations
│   ├── signal_generator.py           # Trading signal generation
│   └── logger.py                     # Logging configuration
│
├── tests/                              # Test files
│   ├── test_models.py                # Model testing
│   ├── test_data_fetching.py         # Data fetching tests
│   └── test_signals.py               # Signal generation tests
│
└── .streamlit/                         # Streamlit configuration
    └── secrets.toml                   # API keys and secrets
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git LFS (for large model files)
- API Keys:
  - [TwelveData](https://twelvedata.com/) (free tier available)
  - Binance API (optional, for fallback)

### One-Line Installation
```bash
git clone https://github.com/muhammadawaislaal/Xauusd-Pred.git && cd Xauusd-Pred && pip install -r requirements.txt && streamlit run app.py
```

## 🛠️ Installation

### Method 1: Standard Installation
```bash
# Clone with Git LFS
git lfs install
git clone https://github.com/muhammadawaislaal/Xauusd-Pred.git
cd Xauusd-Pred

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Decompress model files (if needed)
python -c "import bz2, os; compressed='xauusd_lstm.keras.bz2'; model='xauusd_lstm.keras'; open(model,'wb').write(bz2.decompress(open(compressed,'rb').read()))"

# If model files are Git LFS pointers, download the real files first
git lfs install
git lfs pull

# Run the application
streamlit run app.py
```

### Method 2: Docker Installation
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs && \
    git lfs install

# Clone repository
RUN git clone https://github.com/muhammadawaislaal/Xauusd-Pred.git .
RUN git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## ⚙️ Configuration

### API Keys Setup
Create `.streamlit/secrets.toml`:
```toml
# .streamlit/secrets.toml
TWELVEDATA_API_KEY = "your_twelve_data_api_key"
BINANCE_API_KEY = "your_binance_api_key_optional"
BINANCE_SECRET_KEY = "your_binance_secret_key_optional"
```

### Model Configuration
```python
# Model settings in app.py
SEQ_LEN = 60                    # Sequence length for LSTM
FORECAST_HORIZON = 1            # Predict 1 step ahead (20 minutes)
PIP_VALUE = {
    'XAU/USD': 0.1,            # 0.1 USD per pip for Gold
    'ETH/USD': 1.0             # 1.0 USD per pip for Ethereum
}
```

### Trading Parameters
```python
LOT_SIZE = {
    'XAU/USD': 0.02,           # Standard lot size for Gold
    'ETH/USD': 0.10            # Standard lot size for Ethereum
}

STOP_LOSS_PIPS = {
    'XAU/USD': 50,             # 50 pips stop loss for Gold
    'ETH/USD': 15              # 15 pips stop loss for Ethereum
}

TAKE_PROFIT_PIPS = {
    'XAU/USD': 125,            # 125 pips take profit for Gold
    'ETH/USD': 40              # 40 pips take profit for Ethereum
}
```

## 📊 Usage Guide

### 1. Access Control
1. Enter a user password in the sidebar.
2. The system validates the password, current public IP address, account status, and subscription expiry.
3. Access is granted to the prediction dashboard only when all checks pass.

The administrator access password is `@awaislaal01#$`. It opens the Apex X FX Admin Dashboard, where the administrator can create users with a username, password, and allowed IP address, update credentials, block or unblock accounts, and archive users. Archived records are retained in the database and are not physically deleted. Blocked users receive a subscription-payment message when they try to sign in.

User records are stored only in Supabase. Run [`supabase_schema.sql`](supabase_schema.sql) once in the Supabase SQL Editor. The table keeps only the essential account fields: username, password hash, allowed IP, and status. Blocked and archived records remain available for access control.

For local development, the `.env` file may contain:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=your-publishable-key
ADMIN_PASSWORD=your-admin-password
TWELVEDATA_API_KEY=your-twelve-data-key
```

For Streamlit Cloud, add the equivalent values under App settings -> Secrets. The app accepts `SUPABASE_URL` plus `SUPABASE_SERVICE_ROLE_KEY` (recommended for this server-side admin app), `SUPABASE_KEY`, or the `NEXT_PUBLIC_...` names above. Keep `.env`, database credentials, and user data private. Never expose a service role key in browser code.

Important for Streamlit Community Cloud: all authentication data is stored remotely in Supabase, not in the app filesystem. This keeps user data independent of Streamlit reruns, sleeps, redeployments, and GitHub updates.

Support: `support.apex.x@gmail.com`

### 2. Asset Selection
- **XAU/USD**: Gold vs US Dollar predictions
- **ETH/USD**: Ethereum vs US Dollar predictions

### 3. Running Analysis
#### Manual Analysis:
1. Click "🔄 Run Analysis" button
2. View real-time predictions
3. Check trading signals with Entry/SL/TP levels

#### Scheduled Analysis:
- **Automatic runs**: 5:00 AM & 5:00 PM PKT daily
- **Signal updates**: Every 20 minutes
- **Market hours coverage**: 24/5 for optimal timing

### 4. Interpreting Signals
#### Signal Types:
- **📈 BUY**: Strong upward movement predicted (+1.5+ pips for XAU/USD, +0.2+ for ETH/USD)
- **📉 SELL**: Strong downward movement predicted
- **⏳ WAIT**: Low volatility or unclear direction

#### Signal Details:
- **Entry Price**: Suggested entry point
- **Stop-Loss**: Risk management level
- **Take-Profit**: Profit target level
- **Accuracy**: Model confidence percentage (90-99%)

### 5. Live Charts
- Interactive TradingView charts
- Real-time price updates
- Technical indicator overlays
- Multiple timeframes (5min, 15min, 1hr)

## 🧠 AI Models

### Model Architecture
```python
# Hybrid Model Structure
1. LSTM Network:
   - Input: 60 timesteps × 11 features
   - Layers: 2 LSTM (128 units each)
   - Dropout: 0.2 for regularization
   - Output: Dense layer with linear activation

2. XGBoost Model:
   - Features: Technical indicators + sentiment
   - Parameters: n_estimators=100, max_depth=6
   - Objective: reg:squarederror
```

### Features Used
```
1. Price Features:
   - Open, High, Low, Close
   - Volume

2. Technical Indicators:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - ATR (Average True Range)
   - VWAP (Volume Weighted Average Price)
   - EMA (Exponential Moving Average)
   - ADX (Average Directional Index)
   - CCI (Commodity Channel Index)
   - Stochastic Oscillator
   - OBV (On-Balance Volume)

3. Derived Features:
   - Price momentum
   - Volatility measures
   - Market sentiment
```

### Training Process
1. **Data Collection**: Historical market data
2. **Preprocessing**: Normalization, feature engineering
3. **Model Training**: LSTM and XGBoost training
4. **Validation**: Backtesting with historical data
5. **Deployment**: Model serialization for production

## 📈 Technical Indicators

### Momentum Indicators
- **RSI**: Measures speed and change of price movements
- **MACD**: Trend-following momentum indicator
- **Stochastic Oscillator**: Compares closing price to price range

### Trend Indicators
- **EMA**: Weighted moving average emphasizing recent prices
- **ADX**: Measures trend strength regardless of direction
- **CCI**: Identifies cyclical trends

### Volatility Indicators
- **Bollinger Bands**: Price volatility and relative price levels
- **ATR**: Market volatility by analyzing trading ranges

### Volume Indicators
- **VWAP**: Average price weighted by volume
- **OBV**: Cumulative volume flow indicator

## 💰 Trading Signals

### Signal Generation Logic
```python
def generate_signal(current_price, predicted_price, asset):
    pip_difference = (predicted_price - current_price) / PIP_VALUE[asset]
    
    if pip_difference >= THRESHOLD[asset]:
        return "BUY", current_price, calculate_stop_loss(current_price, "BUY"), calculate_take_profit(current_price, "BUY")
    elif pip_difference <= -THRESHOLD[asset]:
        return "SELL", current_price, calculate_stop_loss(current_price, "SELL"), calculate_take_profit(current_price, "SELL")
    else:
        return "WAIT", None, None, None
```

### Risk Management
- **Position Sizing**: 0.02 lots for XAU/USD, 0.10 lots for ETH/USD
- **Risk/Reward Ratio**: 1:2.5 (Stop Loss: Take Profit)
- **Maximum Risk**: 2% per trade
- **Daily Limit**: Maximum 5 trades per asset

### Performance Metrics
- **Accuracy**: 90-99% based on backtesting
- **Win Rate**: 65-75% historical performance
- **Average Gain**: 15-25 pips per successful trade
- **Maximum Drawdown**: < 20% in stress testing

## 🔒 Security

### Access Control
- **Password Protection**: Valid credentials required
- **IP Validation**: Whitelisted IP addresses
- **Subscription Check**: Active subscription required
- **Time-based Access**: Expiry date validation

### Data Security
- **API Key Encryption**: Secure storage in Streamlit secrets
- **No Sensitive Storage**: API keys not saved to disk
- **Encrypted Communication**: HTTPS for all API calls
- **Local Processing**: Data processed in-memory only

### Privacy Features
- **No User Tracking**: Anonymous usage statistics
- **Session Isolation**: Separate sessions for each user
- **Data Minimization**: Only essential data collected
- **GDPR Compliance**: Privacy-by-design approach

## 🚀 Deployment

### Streamlit Cloud Deployment
```bash
# 1. Prepare repository
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# 2. Deploy via Streamlit Cloud
# - Connect GitHub repository
# - Set secrets in dashboard
# - Deploy main branch
```

### Self-Hosted Deployment
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip nginx git git-lfs

# Clone and setup
git clone https://github.com/muhammadawaislaal/Xauusd-Pred.git
cd Xauusd-Pred
git lfs pull

# Create systemd service
sudo nano /etc/systemd/system/xauusd-pred.service

# Service file:
[Unit]
Description=XAU/USD Prediction Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/xauusd-pred
ExecStart=/usr/bin/streamlit run app.py --server.port=8501 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable xauusd-pred
sudo systemctl start xauusd-pred
```

### Docker Compose Deployment
```yaml
version: '3.8'
services:
  xauusd-pred:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TWELVEDATA_API_KEY=${TWELVEDATA_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone
git clone https://github.com/your-username/Xauusd-Pred.git
cd Xauusd-Pred

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run development server
streamlit run app.py --server.runOnSave true
```

### Contribution Areas
- 🐛 Bug fixes and performance improvements
- 📊 Additional technical indicators
- 🤖 New AI model architectures
- 🌐 Additional asset support (BTC, forex pairs)
- 📱 Mobile-responsive UI enhancements
- 📚 Documentation improvements

### Code Standards
- Follow PEP 8 style guide
- Add type hints for functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update requirements.txt for new dependencies

## 📄 License

MIT License

Copyright (c) 2025 XAU/USD & ETH/USD Predictor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ⚠️ Disclaimer

### Important Notice
This application is for **EDUCATIONAL PURPOSES ONLY**. The predictions and signals generated are based on historical data and AI models, which may not accurately predict future market movements.

### Risk Disclosure
- **Trading involves substantial risk** of loss
- **Past performance does not guarantee future results**
- **You may lose more than your initial investment**
- **Only trade with money you can afford to lose**

### Professional Advice
- Consult with a qualified financial advisor before making investment decisions
- Conduct your own research and due diligence
- Understand the risks involved in forex and cryptocurrency trading
- Never invest based solely on automated signals

### No Financial Advice
The creators and contributors of this project are not financial advisors. This software does not provide financial advice. All trading decisions are the sole responsibility of the user.

## 👨‍💻 Developer

### Project Maintainer
**Muhammad Awais Laal**
- 👨‍💻 AI & Trading Systems Developer
- 📧 Email: m.awaislaal@gmail.com
- 🔗 GitHub: [@muhammadawaislaal](https://github.com/muhammadawaislaal)
- 💼 LinkedIn: [Muhammad Awais Laal](https://linkedin.com/in/muhammadawaislaal)

### Technical Stack
- **Frontend**: Streamlit, Plotly, TradingView Widgets
- **Backend**: Python, TensorFlow, XGBoost
- **Data Processing**: Pandas, NumPy, TA-Lib
- **APIs**: TwelveData, Binance, Yahoo Finance
- **Deployment**: Streamlit Cloud, Docker, AWS
- **Monitoring**: Logging, Performance Metrics

### Support
For technical support, bug reports, or feature requests:
1. Check existing [Issues](https://github.com/muhammadawaislaal/Xauusd-Pred/issues)
2. Create new issue with detailed description
3. Email: support.apex.x@gmail.com

<div align="center">

---

### ⭐ Support the Project

If you find this project useful, please give it a star on GitHub!

**Built with ❤️ for the Trading Community**

*"Educating traders with AI-powered insights"*

</div>
