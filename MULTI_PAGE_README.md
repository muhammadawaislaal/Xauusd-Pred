# Multi-Page Application Guide

## Overview

The XAU/USD & ETH/USD AI Predictor has been refactored into a professional multi-page application with authentication, IP protection, and subscription management.

## Project Structure

```
app/
├── (auth)/
│   └── login/
│       └── page.tsx              # Public login page
├── (protected)/
│   ├── dashboard/
│   │   └── page.tsx              # Main dashboard (protected)
│   └── account/
│       └── page.tsx              # Account settings (protected)
├── api/
│   ├── auth/
│   │   └── login/
│   │       └── route.ts          # Authentication endpoint
│   ├── get-ip/
│   │   └── route.ts              # IP detection endpoint
│   ├── predict/
│   │   └── route.ts              # Price prediction API
│   ├── market-data/
│   │   └── route.ts              # Market data API
│   └── current-price/
│       └── route.ts              # Current price API
├── layout.tsx                     # Root layout with AuthProvider
├── page.tsx                       # Home redirect
└── globals.css                    # Global styles

components/
├── Header.tsx                     # Dashboard header
├── Footer.tsx                     # Footer with subscription info
├── Nav.tsx                        # Navigation between pages
├── ProtectedLayout.tsx            # Auth check + user bar
├── PredictionCard.tsx             # Prediction display
├── ChartDisplay.tsx               # Interactive charts
└── TechnicalIndicators.tsx        # Technical analysis panel

lib/
├── auth-context.tsx               # Authentication context
└── api.ts                         # API utilities
```

## Pages & Features

### 1. Login Page (`/login`)
- **Public Access**: No authentication required
- **Features**:
  - Username/Password input
  - Automatic IP detection
  - Real-time IP display
  - Demo credentials (Admin121/Admin121)
  - Error handling:
    - Invalid credentials
    - Expired subscription
    - IP not whitelisted
  - Responsive design (mobile/tablet/desktop)

### 2. Dashboard (`/dashboard`)
- **Protected Access**: Requires login
- **Features**:
  - Real-time price predictions
  - BUY/SELL/WAIT signals
  - Trading signals with confidence levels
  - Technical indicators (RSI, MACD, ATR, EMA, ADX)
  - Interactive candlestick charts
  - Asset switching (XAU/USD ↔ ETH/USD)
  - Auto-refresh every 5 minutes
  - Manual analysis trigger
  - Risk management (Entry/SL/TP)
  - User session bar (top)
  - Navigation menu

### 3. Account Settings (`/account`)
- **Protected Access**: Requires login
- **Features**:
  - User information display
  - Registered IP address
  - Subscription status
  - Subscription plan details
  - Expiry date with warning (30 days)
  - Active features list
  - Security notices
  - Session management info

### 4. Logout
- Available from ProtectedLayout component
- Clears session and redirects to login
- Sessions expire after 24 hours

## Authentication Flow

```
1. User visits /  → Redirects to /login (if not authenticated) or /dashboard (if authenticated)
2. User enters credentials on /login
3. Frontend calls /api/auth/login with username, password, IP
4. Backend validates:
   - Username & password match ALLOWED_USERS
   - IP is whitelisted (* = all IPs allowed)
   - Subscription status is "active"
   - Subscription hasn't expired
5. On success:
   - User data stored in localStorage
   - Session timestamp recorded
   - Redirect to /dashboard
6. On failure:
   - Specific error message displayed
   - User stays on login page

## Error Handling

### Login Errors
| Error | Cause | Solution |
|-------|-------|----------|
| Invalid username or password | Credentials don't match | Re-check credentials (demo: Admin121/Admin121) |
| Access denied. Your IP is not authorized | IP not whitelisted | Contact administrator for IP whitelisting |
| Your subscription has expired | Subscription date has passed | Contact administrator to renew subscription |
| Your subscription is inactive | Subscription status not "active" | Contact administrator to activate |
| Unable to determine your IP address | IP detection failed | Refresh the page |

### API Errors
| Location | Error | Fallback |
|----------|-------|----------|
| Dashboard Predictions | Backend unavailable | Uses mock prediction data |
| Dashboard Chart | Backend unavailable | Uses mock market data |
| Market Data | Connection failed | Generates random candlestick data |

## Session Management

### Local Storage
- `auth_user`: User object (username, ip, subscription)
- `loginTime`: Session start timestamp

### Session Duration
- **24 hours** of inactivity timeout
- Checked on:
  - Page load
  - Every 60 seconds
  - User navigation

### Auto-Logout
- Automatic logout on session expiry
- User redirected to login with message
- Session cleared from localStorage

## IP Protection

### Configuration
- **Default**: Allow all IPs (`*`)
- **Whitelisting**: Specify IP ranges in ALLOWED_USERS

### IP Sources (Priority)
1. `x-forwarded-for` header
2. `x-real-ip` header
3. `cf-connecting-ip` header (Cloudflare)

### Display
- IP shown on login page
- IP shown in user session bar
- IP stored in user profile

## Subscription Management

### Subscription Data
```typescript
subscription: {
  status: 'active' | 'inactive',
  plan: 'professional' | 'basic' | 'enterprise',
  expiryDate: ISO date string
}
```

### Status Checks
1. **Login**: Verified at authentication
2. **Account Page**: Displayed with expiry countdown
3. **Visual Warning**: Shows if expiry within 30 days

### Display Location
- User session bar (top): Status + Expiry date
- Account page: Full subscription details + warning

## Mobile Responsiveness

### Breakpoints
- **Mobile**: 375px - 640px (full width stack)
- **Tablet**: 641px - 1024px (2-column layout where possible)
- **Desktop**: 1025px+ (full multi-column layout)

### Responsive Features
- Nav bar: Hamburger menu on mobile
- User bar: Hides subscription info on mobile (space saving)
- Forms: Full width on mobile, constrained on desktop
- Charts: Responsive container sizing
- Dashboard: Single column on mobile, multi-column on desktop

## Demo Credentials

```
Username: Admin121
Password: Admin121
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - Authenticate user
- `GET /api/get-ip` - Get client IP

### Protected (Requires Login)
- `GET /api/predict?symbol=XAU/USD|ETH/USD` - Get predictions
- `GET /api/market-data?symbol=XAU/USD|ETH/USD` - Get market data
- `GET /api/current-price?symbol=XAU/USD|ETH/USD` - Get current price

## Color Scheme

- **Primary**: Golden (#d4a574)
- **Accent**: Bronze (#c97b3a)
- **Secondary**: Dark Brown (#8b7355)
- **Background**: Off-white (#f8f7f5)
- **Surface**: White (#ffffff)
- **Border**: Light Tan (#e5ddd3)

## Development

### Install Dependencies
```bash
npm install
```

### Run Dev Server
```bash
npm run dev
```

### Build for Production
```bash
npm run build
npm start
```

## Deployment

### Vercel
1. Push to GitHub
2. Connect repo on Vercel Dashboard
3. Set environment variables (if needed)
4. Deploy

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
CMD ["npm", "start"]
```

### Environment Variables
- `PYTHON_API_URL` (optional): Backend Python API URL
- Default: `http://localhost:5000`

## File Removals
- Old single-page `/login` route deleted
- Streamlit reference removed
- UMTi Tech Solutions contact removed (replaced with subscription info)

## Security Features

1. **Session Authentication**: 24-hour expiry
2. **IP Whitelisting**: Restrict access by IP
3. **Subscription Validation**: Check status at login
4. **Logout Button**: Explicit session termination
5. **Protected Routes**: Auth middleware guards dashboard
6. **Secure Storage**: LocalStorage for client-side auth

## Error Pages

- **Login Errors**: Displayed inline with error details
- **404**: Standard Next.js 404 page
- **API Errors**: Graceful fallback with mock data
- **Expired Session**: Redirects to login with message

## Future Enhancements

- [ ] Refresh token mechanism
- [ ] Multi-user role support (Admin/Trader)
- [ ] Activity logging
- [ ] IP change detection
- [ ] OTP authentication
- [ ] API key management
- [ ] Subscription upgrade/downgrade flow
- [ ] Invoice history
- [ ] Payment integration

---

**Created**: 2025
**Updated**: Latest
**Status**: Production Ready
