# XAU/USD & ETH/USD AI Predictor - Professional Frontend

A modern, professional Next.js frontend for the XAU/USD & ETH/USD AI prediction system with real-time trading signals, technical analysis, and risk management tools.

## Features

- **Professional Dashboard** - Clean, modern UI with golden/bronze color scheme (no navy/blue)
- **Real-time Predictions** - Live AI predictions for Gold (XAU/USD) and Ethereum (ETH/USD)
- **Trading Signals** - BUY/SELL/WAIT recommendations with confidence scores
- **Risk Management** - Entry, Stop Loss, and Take Profit levels for each signal
- **Technical Analysis** - Real-time display of RSI, MACD, ATR, EMA, ADX indicators
- **Interactive Charts** - Candlestick and volume analysis with Recharts
- **Auto-Refresh** - Automatic updates every 5 minutes
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- **Developer Attribution** - Professional footer with portfolio link on hover

## Tech Stack

- **Frontend Framework**: Next.js 16 with React 19
- **Styling**: Tailwind CSS v4 with custom theme
- **Charts**: Recharts for interactive data visualization
- **HTTP Client**: Axios for API communication
- **Backend Integration**: REST API bridge to Python models

## Installation

### Prerequisites
- Node.js 18+
- npm or yarn
- Python 3.9+ (for backend API server)

### Setup Frontend

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Update .env.local with your Python API URL
# NEXT_PUBLIC_API_URL=http://localhost:5000 (for local development)
# PYTHON_API_URL=http://localhost:5000
```

### Setup Backend API Server

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run Flask API server (on port 5000)
python api_server.py

# Or use Gunicorn for production
gunicorn api_server:app -w 4 -b 0.0.0.0:5000
```

### Run Development Server

```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start Next.js dev server
npm run dev

# Open http://localhost:3000 in your browser
```

## Build for Production

```bash
# Build optimized Next.js app
npm run build

# Start production server
npm start
```

## Deployment to Vercel

### Option 1: Using Vercel Dashboard

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your GitHub repository
4. Set environment variables:
   - `PYTHON_API_URL`: Your backend API URL (e.g., https://api.example.com)
5. Deploy

### Option 2: Using Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Set environment variables when prompted
# PYTHON_API_URL: Your backend API URL
```

## Environment Variables

Create a `.env.local` file with:

```
# Frontend - Public API endpoint
NEXT_PUBLIC_API_URL=http://localhost:5000

# Next.js server-side API endpoint
PYTHON_API_URL=http://localhost:5000

# Optional: TwelveData API Key
# TWELVEDATA_API_KEY=your_api_key_here
```

## Project Structure

```
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main dashboard page
│   ├── globals.css         # Global styles and Tailwind config
│   └── api/
│       ├── predict/        # Prediction endpoint
│       ├── market-data/    # Market data endpoint
│       └── current-price/  # Current price endpoint
├── components/
│   ├── Header.tsx          # Navigation header with asset switcher
│   ├── PredictionCard.tsx  # Prediction and signal display
│   ├── ChartDisplay.tsx    # Interactive price charts
│   ├── TechnicalIndicators.tsx # Technical analysis dashboard
│   └── Footer.tsx          # Footer with developer attribution
├── lib/
│   └── api.ts             # API client utilities
├── public/                 # Static assets
├── api_server.py          # Flask API bridge to Python models
├── package.json           # Dependencies
├── tsconfig.json          # TypeScript config
├── tailwind.config.js     # Tailwind theme customization
├── postcss.config.js      # PostCSS configuration
└── vercel.json            # Vercel deployment config
```

## API Endpoints

The Next.js API routes bridge to the Python Flask backend:

- `GET /api/predict?symbol=XAU/USD` - Get AI prediction
- `GET /api/market-data?symbol=XAU/USD` - Get historical market data
- `GET /api/current-price?symbol=XAU/USD` - Get current price

All endpoints accept `symbol` parameter: `XAU/USD` or `ETH/USD`

## Color Scheme

The app uses a professional golden/bronze theme (no navy/blue):

- **Primary**: `#d4a574` (Warm gold)
- **Secondary**: `#8b7355` (Warm brown)
- **Accent**: `#c97b3a` (Deep gold)
- **Background**: `#f8f7f5` (Cream white)
- **Surface**: `#ffffff` (White)
- **Border**: `#e5ddd3` (Light taupe)

## Features

### Dashboard Controls
- **Run Analysis** - Manually trigger AI prediction and analysis
- **Auto-Refresh Toggle** - Enable/disable automatic updates (5-minute interval)
- **Asset Switcher** - Toggle between XAU/USD (Gold) and ETH/USD (Ethereum)

### Prediction Display
- Current market price
- AI-predicted price (20 minutes ahead)
- Trading signal with confidence level
- Pip difference calculation
- Price change percentage

### Risk Management
- **Entry Price** - Recommended entry point
- **Stop Loss** - Risk management level (recommended max loss)
- **Take Profit** - Profit target level
- **Risk/Reward Ratio** - 1:2.5 (industry standard)

### Technical Indicators
Real-time display with status indicators:
- **RSI** - Relative Strength Index (momentum)
- **MACD** - Moving Average Convergence Divergence (trend)
- **ATR** - Average True Range (volatility)
- **EMA** - Exponential Moving Average (trend direction)
- **ADX** - Average Directional Index (trend strength)

### Interactive Charts
- **Price Action** - OHLC (Open, High, Low, Close) candlesticks
- **Volume** - Trading volume over time
- **Indicators** - Technical overlays (High, Low bands)
- **Responsive** - Adapts to all screen sizes
- **Multiple Timeframes** - 5min, 15min, hourly data

### Developer Attribution
- Professional footer with developer name
- Hover tooltip linking to portfolio: https://muhammadawaislaal.github.io/My_PortFolio/
- Email contact for support
- GitHub repository link
- Copyright and disclaimer notices

## Troubleshooting

### "No prediction data available" message
- Ensure the Python Flask API server is running (`python api_server.py`)
- Check that the API URL in environment variables is correct
- The app will fall back to mock data if API is unavailable

### Build errors with Tailwind
- The project uses Tailwind CSS v4 with new configuration format
- Ensure you're using the correct PostCSS config
- Try: `npm install @tailwindcss/postcss --save-dev`

### Port conflicts
- Frontend: Default port 3000 (change with `npm run dev -- -p 3001`)
- Backend API: Default port 5000 (change in `api_server.py`)

## Performance Tips

1. **Enable Auto-Refresh** for live trading insights
2. **Run Analysis** before market hours for best predictions
3. **Use Stop Loss** - Always protect your capital
4. **Monitor Indicators** - RSI >70 = Overbought, <30 = Oversold

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Keep TwelveData API key private
- The frontend does NOT handle user authentication (server-side only)

## Support

For technical issues:
- Email: m.awaislaal@gmail.com
- Email: umtitechsolutions@gmail.com
- GitHub: https://github.com/muhammadawaislaal/Xauusd-Pred

## Disclaimer

This is an **educational tool** for learning AI-powered price prediction. It is **NOT financial advice**. Always:
- Conduct your own research
- Consult with qualified financial advisors
- Only trade with money you can afford to lose
- Understand the risks involved in trading

## License

MIT License - See repository for details

---

**Developed by Muhammad Awais Laal**  
[Portfolio](https://muhammadawaislaal.github.io/My_PortFolio/) • [GitHub](https://github.com/muhammadawaislaal)
