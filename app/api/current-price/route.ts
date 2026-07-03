import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol') || 'XAU/USD';

  if (!['XAU/USD', 'ETH/USD'].includes(symbol)) {
    return NextResponse.json({ error: 'Invalid symbol' }, { status: 400 });
  }

  try {
    const apiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000';
    const response = await fetch(`${apiUrl}/api/current-price?symbol=${symbol}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Current price route error:', error);
    return NextResponse.json({
      price: symbol === 'XAU/USD' ? 2450.5 : 2650.75,
      timestamp: new Date().toISOString(),
    });
  }
}
