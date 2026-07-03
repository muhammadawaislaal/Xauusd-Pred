# Deployment Guide - XAU/USD & ETH/USD AI Predictor

This guide explains how to deploy the professional frontend to Vercel and connect it with your Python backend.

## Quick Start Deployment (5 minutes)

### Prerequisites
- GitHub account with your repository
- Vercel account (free at https://vercel.com)
- Python backend running (local or cloud)

### Step 1: Push to GitHub

```bash
cd /path/to/Xauusd-Pred
git add .
git commit -m "Add Next.js frontend"
git push origin main
```

### Step 2: Deploy to Vercel

**Option A: Using Vercel Dashboard (Easiest)**

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Framework: Select "Next.js"
4. Environment Variables:
   ```
   PYTHON_API_URL=https://your-backend-api.com
   ```
5. Click Deploy

**Option B: Using Vercel CLI**

```bash
npm install -g vercel
vercel
# Follow prompts and set environment variable PYTHON_API_URL
```

### Step 3: Configure Backend API URL

After deployment, your site is live at `https://your-project.vercel.app`

Update the environment variable:
1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add `PYTHON_API_URL=https://your-backend-api.com` (your Python API endpoint)
3. Redeploy the project

## Detailed Setup Instructions

### Local Development

#### 1. Frontend Development Server

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# App runs at http://localhost:3000
```

#### 2. Backend API Server

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Flask API
python api_server.py

# API runs at http://localhost:5000
```

#### 3. Verify Connection

In your browser:
- Frontend: http://localhost:3000
- Click "Run Analysis" button
- Should see BUY/SELL signal with prices
- If API fails, fallback mock data shows

### Production Deployment

#### Option 1: Vercel Frontend + Vercel Serverless Backend

**Deploy Frontend to Vercel:**

```bash
# Connected GitHub repository auto-deploys on push
git push origin main
```

**Deploy Backend (Python API) to Vercel:**

Create `api/predict.py`:

```python
# Use Vercel's Python runtime
from flask import Flask
# ... import your models ...

app = Flask(__name__)

@app.route('/api/predict')
def predict():
    # Your prediction logic
    pass

# Vercel will auto-detect this as serverless function
```

Or use environment variables:
```
PYTHON_API_URL=https://your-vercel-backend.com
```

#### Option 2: Vercel Frontend + External Backend

**Vercel Frontend:**
1. Deploy as above
2. Set `PYTHON_API_URL` to your backend URL

**Backend Options:**
- Render.com (free tier available)
- Railway.app
- PythonAnywhere
- AWS EC2
- DigitalOcean
- Azure App Service
- Google Cloud Run
- Heroku (deprecated)

Example for Render.com:

```bash
# Create Render service
# Set build command: pip install -r requirements.txt
# Set start command: gunicorn api_server:app -w 4 -b 0.0.0.0:5000
# Add TWELVEDATA_API_KEY env var
# Deploy Dockerfile or connect GitHub
```

#### Option 3: Docker Containerization

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "api_server:app", "-w", "4", "-b", "0.0.0.0:5000"]
```

Deploy to:
- Docker Hub
- AWS ECR
- Google Container Registry
- Azure Container Registry

### Environment Variables

**Required:**
- `PYTHON_API_URL` - Backend API endpoint URL

**Optional:**
- `TWELVEDATA_API_KEY` - For price data fetching (in backend)
- `NODE_ENV` - Set to "production" on Vercel

### Vercel Configuration

`vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "framework": "nextjs",
  "nodeVersion": "20.x"
}
```

### Troubleshooting Deployment

**Problem: "No prediction data available"**
- Check `PYTHON_API_URL` is set correctly
- Verify backend API is running and accessible
- Check CORS is enabled on backend (Flask-CORS)
- Look at browser console for error details

**Problem: Build fails**
```bash
# Clear cache and rebuild
rm -rf .next
npm run build
```

**Problem: API timeout**
- Frontend waits 30 seconds for prediction
- If backend slower, increase timeout in `app/page.tsx`
- Optimize backend model loading

**Problem: Environment variables not loading**
- Redeploy after adding env vars
- Use `vercel env list` to verify
- Check `.env.local` is in `.gitignore`

### Performance Optimization

**Frontend:**
- Next.js automatically optimizes bundles
- Charts lazy-load on interaction
- API calls cache for 5 minutes

**Backend:**
- Load models once on startup
- Implement caching for predictions
- Use Gunicorn workers (4-8 workers)

### CORS Configuration

If backend and frontend on different domains:

In `api_server.py`:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-frontend.vercel.app"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

### SSL/HTTPS

- Vercel provides free SSL certificates
- Backend on Render/Railway also includes SSL
- Ensure API URL uses HTTPS in production

## Monitoring & Logging

### Monitor Deployments

**Vercel Dashboard:**
- Real-time logs
- Performance metrics
- Error tracking

**Backend Logging:**

In `api_server.py`:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction: {symbol} - Signal: {signal}")
```

### Error Tracking

Add Sentry for production errors:

```bash
npm install @sentry/nextjs
# Configure in next.config.js
```

## Security Best Practices

1. **Never commit secrets:**
   ```bash
   echo ".env.local" >> .gitignore
   echo ".env.*.local" >> .gitignore
   ```

2. **Use environment variables** for sensitive data

3. **Enable rate limiting** on backend:
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=lambda: request.remote_addr)
   
   @app.route('/api/predict')
   @limiter.limit("30/hour")
   def predict():
       pass
   ```

4. **Validate API requests:**
   ```python
   from zod import BaseModel, validator
   
   class PredictRequest(BaseModel):
       symbol: str
       
       @validator('symbol')
       def symbol_valid(cls, v):
           if v not in ['XAU/USD', 'ETH/USD']:
               raise ValueError('Invalid symbol')
           return v
   ```

5. **Set CORS properly** (don't allow all origins)

## Scaling Considerations

### High Traffic

1. **Frontend:** Vercel auto-scales
2. **Backend:** 
   - Use load balancer (AWS ALB, Nginx)
   - Deploy multiple instances
   - Use Redis for model caching
   - Consider serverless (AWS Lambda, Google Cloud Functions)

### Database Optimization

If adding database:
- Use connection pooling
- Cache frequently accessed data
- Index prediction results by symbol

### Model Optimization

- Quantize LSTM models (reduces size)
- Use ONNX for faster inference
- Implement prediction batching
- Cache recent predictions (5-minute window)

## Maintenance

### Update Dependencies

```bash
npm update
pip install --upgrade -r requirements.txt
```

### Monitor Costs

- **Vercel:** Free tier includes 100GB bandwidth/month
- **Render/Railway:** Check usage to avoid overages
- **TwelveData API:** Free tier: 800 requests/day

### Regular Backups

- GitHub automatically backs up code
- If using database, enable automatic backups

## Rollback & Disaster Recovery

**Vercel:**
1. Dashboard → Deployments
2. Select previous deployment
3. Click "Redeploy"

**Git:**
```bash
git revert HEAD
git push origin main
```

## Support & Resources

- **Vercel Docs:** https://vercel.com/docs
- **Next.js:** https://nextjs.org/docs
- **Flask:** https://flask.palletsprojects.com
- **TwelveData API:** https://twelvedata.com/docs

## Questions?

Email: m.awaislaal@gmail.com or umtitechsolutions@gmail.com

---

**Deployed Successfully!** Your AI Predictor frontend is now live and ready to serve trading signals worldwide.
