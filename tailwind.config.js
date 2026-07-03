/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0f',
        surface: '#14141e',
        border: '#2a2a3a',
        'text-primary': '#ffffff',
        'text-muted': '#94a3b8',
        'accent-primary': '#7c3aed',
        'accent-secondary': '#3b82f6',
        'signal-buy': '#34d399',
        'signal-sell': '#f43f5e',
        'signal-wait': '#fbbf24',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      borderRadius: {
        xl: '12px',
      },
      boxShadow: {
        'glow-purple': '0 0 24px rgba(124, 58, 237, 0.5)',
        'glow-blue': '0 0 24px rgba(59, 130, 246, 0.5)',
      },
      animation: {
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
