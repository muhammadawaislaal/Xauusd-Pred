'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';

const navItems = [
  { label: 'Dashboard', href: '/dashboard', icon: '📊' },
  { label: 'Account', href: '/account', icon: '👤' },
  { label: 'History', href: '#', icon: '📈' },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-50 lg:hidden p-2 rounded-lg bg-card-bg border border-border hover:border-border-light transition-all"
      >
        <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isOpen ? 'M6 18L18 6M6 6l12 12' : 'M4 6h16M4 12h16M4 18h16'} />
        </svg>
      </button>

      {/* Mobile overlay */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setIsOpen(false)} />
      )}

      {/* Sidebar */}
      <div className={`fixed left-0 top-0 h-screen w-64 bg-card-bg border-r border-border z-40 lg:z-0 transform transition-transform duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}>
        <div className="h-full flex flex-col p-6">
          {/* Logo */}
          <div className="mb-8">
            <Link href="/dashboard" onClick={() => setIsOpen(false)}>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                  <span className="text-lg font-bold text-white">⚡</span>
                </div>
                <h1 className="text-xl font-bold text-foreground">TradeAI</h1>
              </div>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                  pathname === item.href
                    ? 'bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 text-foreground'
                    : 'text-secondary hover:text-foreground hover:bg-border/50'
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </Link>
            ))}
          </nav>

          {/* User section */}
          {user && (
            <div className="space-y-4 border-t border-border pt-6">
              <div className="flex items-center gap-3 px-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-white">{user.username[0].toUpperCase()}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{user.username}</p>
                  <p className="text-xs text-secondary truncate">{user.ip}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-all font-medium text-sm"
              >
                <span>🚪</span>
                <span>Logout</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Sidebar spacer for desktop */}
      <div className="hidden lg:block w-64" />
    </>
  );
}
