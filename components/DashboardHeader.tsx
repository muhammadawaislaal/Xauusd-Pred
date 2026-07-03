'use client';

import { useAuth } from '@/lib/auth-context';
import { useRouter } from 'next/navigation';
import { LogOut, Settings, BarChart3 } from 'lucide-react';
import Link from 'next/link';

export function DashboardHeader() {
  const { user, logout } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  return (
    <header className="bg-surface border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="inline-flex items-center justify-center w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold text-foreground">AI Predictor</h1>
              <p className="text-xs text-secondary">XAU/USD & ETH/USD</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            <Link
              href="/dashboard"
              className="text-foreground hover:text-primary transition-colors font-medium"
            >
              Dashboard
            </Link>
            <Link
              href="/account"
              className="text-foreground hover:text-primary transition-colors font-medium"
            >
              Account
            </Link>
            <Link
              href="/settings"
              className="text-foreground hover:text-primary transition-colors font-medium"
            >
              Settings
            </Link>
          </nav>

          {/* User Info & Actions */}
          <div className="flex items-center gap-3 sm:gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-medium text-foreground">{user?.username}</p>
              <p className="text-xs text-secondary">
                {user?.subscription?.plan || 'Active'}
              </p>
            </div>

            <button
              title="Settings"
              className="p-2 hover:bg-secondary/10 rounded-lg transition-colors text-secondary hover:text-primary"
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={handleLogout}
              title="Logout"
              className="p-2 hover:bg-red-50 rounded-lg transition-colors text-secondary hover:text-red-600 md:px-4 md:py-2 md:bg-transparent md:hover:bg-red-50 md:flex md:items-center md:gap-2"
            >
              <LogOut className="w-5 h-5" />
              <span className="hidden md:inline text-sm font-medium">Logout</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
