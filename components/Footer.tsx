'use client';

import React, { useState } from 'react';

export default function Footer() {
  const [showPortfolio, setShowPortfolio] = useState(false);

  return (
    <footer className="border-t border-border bg-surface mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded flex items-center justify-center">
              <span className="text-white font-bold text-sm">AI</span>
            </div>
            <p className="text-sm text-secondary">Professional Trading Intelligence</p>
          </div>

          <div className="text-center flex-1">
            <div
              className="inline-block relative group cursor-pointer"
              onMouseEnter={() => setShowPortfolio(true)}
              onMouseLeave={() => setShowPortfolio(false)}
            >
              <p className="text-sm text-secondary hover:text-primary smooth-transition">
                Developed by <span className="font-semibold text-foreground">Muhammad Awais Laal</span>
              </p>
              {showPortfolio && (
                <div className="absolute left-1/2 transform -translate-x-1/2 bottom-full mb-2 bg-surface border border-border rounded-lg shadow-lg p-2 z-50 whitespace-nowrap">
                  <a
                    href="https://muhammadawaislaal.github.io/My_PortFolio/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:text-accent smooth-transition underline"
                  >
                    View Portfolio
                  </a>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-secondary">
            <a
              href="https://github.com/muhammadawaislaal/Xauusd-Pred"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary smooth-transition"
            >
              GitHub
            </a>
            <span>•</span>
            <a
              href="mailto:m.awaislaal@gmail.com"
              className="hover:text-primary smooth-transition"
            >
              Contact
            </a>
            <span>•</span>
            <p>© 2025 XAU/USD & ETH/USD Predictor</p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-secondary">
            <div>
              <h4 className="font-semibold text-foreground mb-2">Disclaimer</h4>
              <p>This is an educational tool. Not financial advice. Always conduct your own research.</p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-2">Risk Notice</h4>
              <p>Trading involves substantial risk. Past performance does not guarantee future results.</p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-2">Subscription</h4>
              <p>Contact administrator for subscription details, renewals, and account management.</p>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
