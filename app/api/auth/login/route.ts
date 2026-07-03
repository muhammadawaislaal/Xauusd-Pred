import { NextRequest, NextResponse } from 'next/server';

interface LoginRequest {
  username: string;
  password: string;
  ip: string;
}

interface AllowedUser {
  username: string;
  password: string;
  allowed_ips: string[];
  subscription?: {
    status: string;
    expiryDate?: string;
    plan?: string;
  };
}

// Mock allowed users (in production, fetch from your Python backend)
const ALLOWED_USERS: AllowedUser[] = [
  {
    username: 'Admin121',
    password: 'Admin121',
    allowed_ips: ['*'], // Allow all IPs, or specify: ['192.168.1.1', '10.0.0.0/8']
    subscription: {
      status: 'active',
      plan: 'professional',
      expiryDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString(),
    },
  },
];

function isIPAllowed(clientIP: string, allowedIPs: string[]): boolean {
  // If * is in allowed IPs, allow all
  if (allowedIPs.includes('*')) {
    return true;
  }

  // Check exact match
  if (allowedIPs.includes(clientIP)) {
    return true;
  }

  // Check CIDR notation (basic implementation)
  for (const allowedIP of allowedIPs) {
    if (allowedIP.includes('/')) {
      // Simple CIDR check - in production, use a proper library
      const [network, bits] = allowedIP.split('/');
      // For now, just do basic checks
      if (clientIP.startsWith(network.split('.').slice(0, 3).join('.'))) {
        return true;
      }
    }
  }

  return false;
}

export async function POST(request: NextRequest) {
  try {
    const body: LoginRequest = await request.json();
    const { username, password, ip } = body;

    // Validate input
    if (!username || !password || !ip) {
      return NextResponse.json(
        { error: 'Username, password, and IP are required' },
        { status: 400 }
      );
    }

    // Find user
    const user = ALLOWED_USERS.find(u => u.username === username);
    if (!user) {
      return NextResponse.json(
        { error: 'Invalid username or password' },
        { status: 401 }
      );
    }

    // Verify password
    if (user.password !== password) {
      return NextResponse.json(
        { error: 'Invalid username or password' },
        { status: 401 }
      );
    }

    // Verify IP
    if (!isIPAllowed(ip, user.allowed_ips)) {
      return NextResponse.json(
        { error: `Access denied. Your IP (${ip}) is not authorized` },
        { status: 403 }
      );
    }

    // Check subscription status
    if (user.subscription?.status !== 'active') {
      return NextResponse.json(
        { error: 'Your subscription is inactive. Please renew your subscription.' },
        { status: 403 }
      );
    }

    // Check subscription expiry
    if (user.subscription?.expiryDate) {
      const expiryDate = new Date(user.subscription.expiryDate);
      if (expiryDate < new Date()) {
        return NextResponse.json(
          { error: 'Your subscription has expired. Please renew to continue.' },
          { status: 403 }
        );
      }
    }

    // Successful login
    return NextResponse.json({
      success: true,
      user: {
        username: user.username,
        ip,
        subscription: user.subscription,
      },
    });
  } catch (error) {
    console.error('[v0] Login error:', error);
    return NextResponse.json(
      { error: 'An error occurred during login. Please try again.' },
      { status: 500 }
    );
  }
}
