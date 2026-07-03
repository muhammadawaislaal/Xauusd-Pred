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
      <div className="bg-sidebar-bg text-surface px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between text-sm sticky top-0 z-40 shadow-md">
        <div className="flex items-center gap-4 sm:gap-6 flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-accent/30 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-accent" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="min-w-0">
              <p className="font-semibold text-surface truncate">{user.username}</p>
              <p className="text-xs text-surface/70 truncate">{user.ip}</p>
            </div>
          </div>
          {user.subscription && (
            <div className="hidden sm:block border-l border-surface/20 pl-4">
              <p className="text-xs text-surface/70">Status: <span className="font-semibold text-accent capitalize">{user.subscription.status}</span></p>
              <p className="text-xs text-surface/70 mt-0.5">Expires: {new Date(user.subscription.expiryDate || '').toLocaleDateString()}</p>
            </div>
          )}
        </div>
        <button
          onClick={handleLogout}
          className="bg-surface/20 hover:bg-surface/30 px-3 sm:px-4 py-1.5 rounded-lg font-medium text-xs sm:text-sm transition-colors whitespace-nowrap ml-4 flex-shrink-0"
        >
          Logout
        </button>
      </div>

      {/* Main content */}
      {children}
    </div>
  );
}
