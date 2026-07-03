'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { Menu, X, LogOut, BarChart3, Settings, History } from 'lucide-react'

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()
  const router = useRouter()

  const navItems = [
    { label: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { label: 'Account', href: '/account', icon: Settings },
    { label: 'History', href: '/history', icon: History },
  ]

  const isActive = (href: string) => pathname === href

  const handleLogout = () => {
    router.push('/login')
  }

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden fixed top-4 left-4 z-50 p-2 bg-surface border border-border rounded-lg text-text-primary hover:bg-background transition"
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setIsOpen(false)}
        ></div>
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-60 bg-surface border-r border-border flex flex-col transition-transform duration-300 z-40 md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
              <span className="text-white font-bold text-lg">⚡</span>
            </div>
            <div>
              <h1 className="text-text-primary font-bold text-lg">Trading</h1>
              <p className="text-text-muted text-xs">Signals</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map(({ label, href, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              onClick={() => setIsOpen(false)}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition ${
                isActive(href)
                  ? 'bg-gradient-to-r from-accent-primary/20 to-accent-secondary/20 border border-accent-primary/30 text-text-primary'
                  : 'text-text-muted hover:bg-background hover:text-text-primary'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{label}</span>
            </Link>
          ))}
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-border space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-text-primary text-sm font-semibold truncate">Trader</p>
              <p className="text-text-muted text-xs truncate">trader@example.com</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-background border border-border rounded-lg text-text-primary hover:bg-signal-sell/10 hover:border-signal-sell/30 transition text-sm font-medium"
          >
            <LogOut size={18} />
            Logout
          </button>
        </div>
      </aside>

      {/* Main content spacer for desktop */}
      <div className="hidden md:block w-60 flex-shrink-0"></div>
    </>
  )
}
