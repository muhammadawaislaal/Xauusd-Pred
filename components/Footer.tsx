'use client';

import React, { useState } from 'react';

export default function Footer() {
  const [showPortfolio, setShowPortfolio] = useState(false);

  return (
    <footer className="border-t border-border bg-surface mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-surface" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V15a1 1 0 11-2 0V5.414L5.707 10.707a1 1 0 01-1.414-1.414l6-6z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="text-sm text-secondary">Premium Trading Intelligence Platform</p>
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
                <div className="absolute left-1/2 transform -translate-x-1/2 bottom-full mb-2 bg-surface border border-border rounded-lg shadow-lg p-3 z-50 whitespace-nowrap">
                  <a
                    href="https://muhammadawaislaal.github.io/My_PortFolio/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:text-accent smooth-transition underline font-semibold"
                  >
                    View Developer Portfolio
                  </a>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-secondary">
            <a
              href="mailto:contact@tradingplatform.com"
              className="hover:text-primary smooth-transition"
            >
              Support
            </a>
            <span>•</span>
            <p>© 2025 Trading Predictor</p>
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
