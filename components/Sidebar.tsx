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
        className="md:hidden fixed top-4 left-4 z-50 p-2 bg-white border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 transition"
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 md:hidden"
          onClick={() => setIsOpen(false)}
        ></div>
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-60 bg-white border-r border-slate-300 flex flex-col transition-transform duration-300 z-40 md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo */}
        <div className="p-6 border-b border-slate-300">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold text-lg">📊</span>
            </div>
            <div>
              <h1 className="text-slate-900 font-bold text-lg">Trading</h1>
              <p className="text-slate-600 text-xs">Signals</p>
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
                  ? 'bg-blue-50 border border-blue-300 text-blue-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{label}</span>
            </Link>
          ))}
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-slate-300 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-slate-900 text-sm font-semibold truncate">Trader</p>
              <p className="text-slate-600 text-xs truncate">trader@example.com</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-700 hover:bg-red-50 hover:border-red-300 hover:text-red-700 transition text-sm font-medium"
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
