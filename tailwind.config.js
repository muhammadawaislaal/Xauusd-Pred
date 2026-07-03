/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#f8f7f5',
        foreground: '#1a1a1a',
        primary: '#d4a574',
        secondary: '#8b7355',
        accent: '#c97b3a',
        surface: '#ffffff',
        border: '#e5ddd3',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
