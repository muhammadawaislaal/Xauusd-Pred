'use client';

import { useAuth } from '@/lib/auth-context';
import { useRouter } from 'next/navigation';
import { useEffect, ReactNode } from 'react';

interface ProtectedLayoutProps {
  children: ReactNode;
}

export function ProtectedLayout({ children }: ProtectedLayoutProps) {
  const { isAuthenticated, isLoading, user, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="text-center">
          <div className="inline-block">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          </div>
          <p className="mt-4 text-foreground font-medium">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return null;
  }

  return (
    <div className="relative">
      {/* User session bar */}
      <div className="bg-gradient-to-r from-primary to-accent text-white px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between text-sm sticky top-0 z-40 shadow-md">
        <div className="flex items-center gap-6 flex-1">
          <div>
            <p className="font-semibold text-white">User: <span className="font-mono">{user.username}</span></p>
            <p className="text-xs text-white/80 mt-1">IP: <span className="font-mono">{user.ip}</span></p>
          </div>
          {user.subscription && (
            <div className="hidden md:block border-l border-white/30 pl-6">
              <p className="font-semibold text-white">Subscription: <span className="font-mono capitalize">{user.subscription.status}</span></p>
              <p className="text-xs text-white/80 mt-1">Expires: <span className="font-mono">{new Date(user.subscription.expiryDate || '').toLocaleDateString()}</span></p>
            </div>
          )}
        </div>
        <button
          onClick={handleLogout}
          className="bg-white/20 hover:bg-white/30 px-4 py-2 rounded font-medium text-sm transition-colors whitespace-nowrap ml-4"
        >
          Logout
        </button>
      </div>

      {/* Main content */}
      {children}
    </div>
  );
}
