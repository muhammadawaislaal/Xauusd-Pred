import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // Get IP from various sources
    const forwardedFor = request.headers.get('x-forwarded-for');
    const realIp = request.headers.get('x-real-ip');
    const cfIp = request.headers.get('cf-connecting-ip');
    
    const ip = forwardedFor?.split(',')[0].trim() || realIp || cfIp || 'unknown';

    return NextResponse.json({ ip });
  } catch (error) {
    console.error('[v0] Error getting IP:', error);
    return NextResponse.json({ ip: 'unknown' }, { status: 500 });
  }
}
