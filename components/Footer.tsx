'use client';

import React, { useState } from 'react';

export default function Footer() {
  const [showPortfolio, setShowPortfolio] = useState(false);

  return (
    <footer className="border-t border-border bg-card-bg/50 mt-12 pt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12 pb-12 border-b border-border">
          {/* Brand Section */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                <span className="text-lg font-bold text-white">⚡</span>
              </div>
              <h3 className="text-lg font-bold text-foreground">TradeAI</h3>
            </div>
            <p className="text-sm text-secondary">Advanced AI-powered price predictions and trading signals for XAU/USD and ETH/USD markets.</p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-foreground mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm text-secondary">
              <li>
                <a href="#" className="hover:text-blue-400 transition-colors">Dashboard</a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-400 transition-colors">Account</a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-400 transition-colors">History</a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-400 transition-colors">Support</a>
              </li>
            </ul>
          </div>

          {/* Developer Attribution */}
          <div>
            <h4 className="font-semibold text-foreground mb-4">Developer</h4>
            <div
              className="relative cursor-pointer group"
              onMouseEnter={() => setShowPortfolio(true)}
              onMouseLeave={() => setShowPortfolio(false)}
            >
              <p className="text-sm text-secondary hover:text-blue-400 transition-colors">
                Developed by
                <br />
                <span className="font-semibold text-foreground">Muhammad Awais Laal</span>
              </p>
              {showPortfolio && (
                <div className="absolute right-0 bottom-full mb-3 bg-card-bg border border-border rounded-lg shadow-lg p-3 z-50 whitespace-nowrap card-shadow">
                  <a
                    href="https://muhammadawaislaal.github.io/My_PortFolio/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-400 hover:text-purple-400 transition-colors underline font-semibold block"
                  >
                    View Developer Portfolio →
                  </a>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Footer */}
        <div className="py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-secondary">
          <p>&copy; 2025 TradeAI. All rights reserved.</p>
          <div className="flex items-center gap-4">
            <a href="#" className="hover:text-blue-400 transition-colors">Privacy Policy</a>
            <span className="text-border">•</span>
            <a href="#" className="hover:text-blue-400 transition-colors">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
