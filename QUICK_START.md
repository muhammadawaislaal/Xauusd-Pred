# Quick Start Guide

## Start the Application

### Development Mode
```bash
# Terminal 1: Start the Next.js dev server
npm run dev

# Terminal 2 (Optional): Start Python backend
python api_server.py
```

The app will be available at: `http://localhost:3000`

## Login

1. **Navigate to Login Page**: `http://localhost:3000/login`
2. **Enter Demo Credentials**:
   - Username: `Admin121`
   - Password: `Admin121`
3. **Your IP**: Will auto-detect (e.g., `::1` for localhost)
4. **Click Login**: System will validate and redirect to dashboard

## Explore the App

### Dashboard (`/dashboard`)
- View real-time AI price predictions
- See trading signals (BUY/SELL/WAIT)
- Check technical indicators
- Switch between XAU/USD and ETH/USD
- Run manual analysis
- Toggle auto-refresh (5-minute interval)

### Account (`/account`)
- View your username and registered IP
- Check subscription status
- See expiry date and plan details
- Review active features
- Read security information

### Navigation
- Click **Dashboard** to go to predictions
- Click **Account** to view settings
- Click **Logout** to exit and return to login

## Pages Overview

| Page | URL | Access | Purpose |
|------|-----|--------|---------|
| Login | `/login` | Public | Enter credentials |
| Dashboard | `/dashboard` | Protected | Trading predictions |
| Account | `/account` | Protected | Settings & subscription |
| Home | `/` | Any | Auto-redirect |

## Features

### Authentication
- ✓ Username/Password login
- ✓ IP address detection
- ✓ 24-hour session timeout
- ✓ Auto-logout on expiry

### Trading
- ✓ Real-time predictions
- ✓ Buy/Sell/Wait signals
- ✓ Confidence levels
- ✓ Risk management (Entry/SL/TP)

### Analysis
- ✓ 5 technical indicators (RSI, MACD, ATR, EMA, ADX)
- ✓ Interactive candlestick charts
- ✓ Volume analysis
- ✓ Market data tracking

### Settings
- ✓ Subscription status
- ✓ Expiry monitoring
- ✓ Feature list
- ✓ Security info

## Error Scenarios & Solutions

### "Invalid username or password"
- Check username: `Admin121`
- Check password: `Admin121`
- Ensure no extra spaces

### "Access denied. Your IP is not authorized"
- Default config allows all IPs
- Contact admin if restricted

### "Your subscription has expired"
- Subscription demo expires: 2025-12-31
- Contact admin to renew

### "Unable to determine your IP address"
- Refresh the page
- Check browser network is working

### Dashboard shows yellow warning
- Backend API might be offline
- App uses fallback mock data
- All features still work

## Mobile Testing

### Test on Different Devices
```bash
# Chrome DevTools
1. Open DevTools (F12)
2. Click Device Toolbar (Ctrl+Shift+M)
3. Select device (iPhone, iPad, etc.)
4. Reload page
```

### Responsive Breakpoints
- **Mobile**: 375px - 640px (hamburger menu)
- **Tablet**: 641px - 1024px
- **Desktop**: 1025px+ (full layout)

## Logout

Click the **Logout** button in the top-right corner:
- Clears session
- Redirects to `/login`
- Session removed from storage

## Session Timeout

- **Duration**: 24 hours from login
- **Auto-check**: Every 60 seconds
- **Behavior**: Auto-logout when expired
- **User Experience**: Redirected to login with notice

## Color Theme

- **Primary Golden**: #d4a574
- **Accent Bronze**: #c97b3a
- **Secondary Brown**: #8b7355
- **Background Cream**: #f8f7f5
- **Surface White**: #ffffff

## Development Info

### Technologies
- Next.js 16 (React framework)
- React 19 (UI library)
- Tailwind CSS v4 (Styling)
- TypeScript (Type safety)

### Key Files
- `app/layout.tsx` - Root layout with AuthProvider
- `app/(auth)/login/page.tsx` - Login page
- `app/(protected)/dashboard/page.tsx` - Dashboard
- `app/(protected)/account/page.tsx` - Account settings
- `lib/auth-context.tsx` - Authentication state
- `components/ProtectedLayout.tsx` - Auth guard

### Disable Warnings
You may see these warnings (safe to ignore):
- "Unsupported metadata viewport" - Layout-specific
- "Detected scroll-behavior: smooth" - CSS-related
- "Next.js DevTools" - Browser console message

## Production Deployment

### Vercel (Recommended)
```bash
# 1. Push to GitHub
git push origin main

# 2. On Vercel Dashboard
# - Click "New Project"
# - Select your GitHub repo
# - Click "Deploy"
# - Done! Your app is live
```

### Self-Hosted
```bash
npm run build
npm start
```

### Environment Variables
Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Troubleshooting

### "Cannot find module"
- Run: `npm install`
- Restart dev server

### Port 3000 already in use
- Use different port: `npm run dev -- -p 3001`

### Styles not loading
- Clear: `.next` folder
- Restart: `npm run dev`

### API calls failing
- Check: Is backend running on port 5000?
- Check: Network tab in DevTools
- Check: Backend CORS configuration

## Support

### Documentation
- **IMPLEMENTATION_COMPLETE.md** - Full feature overview
- **MULTI_PAGE_README.md** - Technical details
- **Inline comments** - Throughout the code

### GitHub
- Repository: `muhammadawaislaal/Xauusd-Pred`
- Issues: Report on GitHub Issues
- Discussions: Ask on GitHub Discussions

### Contact
- Email: `m.awaislaal@gmail.com`
- Portfolio: `muhammadawaislaal.github.io/My_PortFolio/`

---

**Status**: Production Ready
**Version**: 2.0 (Multi-Page Application)
**Last Updated**: 2025
