import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Xauusd Prediction Dashboard',
  description: 'AI-powered trading signal platform for XAU/USD and ETH/USD',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  )
}
