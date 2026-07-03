'use client'

import { useEffect } from 'react'

interface TradingViewWidgetProps {
  symbol: 'XAUUSD' | 'ETHUSD'
  height?: string
}

export function TradingViewWidget({ symbol, height = '500px' }: TradingViewWidgetProps) {
  useEffect(() => {
    // Load the TradingView script
    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.async = true
    script.type = 'text/javascript'
    
    // Prepare the widget configuration
    const container = document.getElementById(`tradingview-widget-${symbol}`)
    if (container) {
      // Clear previous content
      container.innerHTML = '<div class="tradingview-widget-container__widget"></div>'
      
      // Create script content
      const scriptContent = {
        width: '100%',
        height,
        symbol: symbol === 'XAUUSD' ? 'OANDA:XAUUSD' : 'BINANCE:ETHUSDT',
        interval: '15',
        timezone: 'Etc/UTC',
        theme: 'light',
        style: '1',
        locale: 'en',
        hideTopToolbar: false,
        hideLegend: false,
        saveImage: true,
        calendar: false,
        hideVolume: false,
      }
      
      // Set the script configuration
      script.textContent = JSON.stringify(scriptContent)
      container.appendChild(script)
    }
  }, [symbol])

  return (
    <div 
      id={`tradingview-widget-${symbol}`} 
      className="w-full rounded-xl overflow-hidden border border-slate-200 shadow-sm"
      style={{ height: 'auto' }}
    >
      <div className="tradingview-widget-container" style={{ height: '500px' }}>
        <div className="tradingview-widget-container__widget"></div>
      </div>
    </div>
  )
}
