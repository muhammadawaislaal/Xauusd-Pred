import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol') || 'XAU/USD';

  if (!['XAU/USD', 'ETH/USD'].includes(symbol)) {
    return NextResponse.json({ error: 'Invalid symbol' }, { status: 400 });
  }

  try {
    // Call the Python API server
    const apiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000';
    const response = await fetch(`${apiUrl}/api/market-data?symbol=${symbol}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Market data route error:', error);
    // Return mock data
    const mockData = {
      timestamp: Array.from({ length: 50 }, (_, i) => {
        const date = new Date();
        date.setMinutes(date.getMinutes() - (50 - i));
        return date.toISOString();
      }),
      open: Array.from({ length: 50 }, () => symbol === 'XAU/USD' ? 2440 + Math.random() * 20 : 2640 + Math.random() * 20),
      high: Array.from({ length: 50 }, () => symbol === 'XAU/USD' ? 2460 + Math.random() * 20 : 2660 + Math.random() * 20),
      low: Array.from({ length: 50 }, () => symbol === 'XAU/USD' ? 2430 + Math.random() * 20 : 2630 + Math.random() * 20),
      close: Array.from({ length: 50 }, () => symbol === 'XAU/USD' ? 2450 + Math.random() * 20 : 2650 + Math.random() * 20),
      volume: Array.from({ length: 50 }, () => 1000000 + Math.random() * 500000),
      rsi: Array.from({ length: 50 }, () => 30 + Math.random() * 40),
      macd: Array.from({ length: 50 }, () => -1 + Math.random() * 2),
    };
    return NextResponse.json(mockData);
  }
}
