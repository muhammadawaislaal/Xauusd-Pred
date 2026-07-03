'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const menuItems = [
    { label: 'Dashboard', href: '/dashboard', icon: '📊' },
    { label: 'Account', href: '/account', icon: '⚙️' },
  ];

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-40 md:hidden bg-primary text-background p-2 rounded-lg"
      >
        ☰
      </button>

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 w-64 bg-surface-dark border-r border-border transition-transform duration-300 z-30
        md:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
              <span className="text-background font-bold">$</span>
            </div>
            <h1 className="text-xl font-bold text-foreground">Predictor</h1>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2">
          {menuItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setIsOpen(false)}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                pathname === item.href
                  ? 'bg-primary text-background font-semibold'
                  : 'text-foreground hover:bg-surface'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>

        {/* User Info & Logout */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-border space-y-3">
          {user && (
            <div className="text-xs text-secondary px-2">
              <p className="font-semibold text-foreground mb-2">Session</p>
              <p className="font-mono text-primary">{user.ip}</p>
              {user.subscription && (
                <p className="text-primary text-opacity-70 mt-2">
                  Expires: {new Date(user.subscription.expiryDate || '').toLocaleDateString()}
                </p>
              )}
            </div>
          )}
          <button
            onClick={handleLogout}
            className="w-full bg-red-500/20 hover:bg-red-500/30 text-red-400 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
          >
            Logout
          </button>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-20 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Main Content Offset */}
      <div className="hidden md:block md:w-64" />
    </>
  );
}
