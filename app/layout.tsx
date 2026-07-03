import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'XAU/USD & ETH/USD AI Predictor',
  description: 'Professional AI-powered price prediction system for Gold and Ethereum with advanced trading signals and technical analysis.',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  openGraph: {
    title: 'XAU/USD & ETH/USD AI Predictor',
    description: 'Professional AI-powered price prediction system',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
      </body>
    </html>
  );
}
