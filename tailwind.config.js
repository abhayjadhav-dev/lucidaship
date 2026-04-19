/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: '#c8a96e',
          dim: 'rgba(200,169,110,0.1)',
          glow: 'rgba(200,169,110,0.35)',
        },
        black: '#080808',
        deep: '#0f0f0f',
        surface: '#141414',
        line: '#222222',
        muted: '#3a3a3a',
        dim: '#888888',
        light: '#c8c8c8',
        white: '#f0ede8',
      },
      fontFamily: {
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"DM Sans"', 'sans-serif'],
        mono: ['"DM Mono"', 'monospace'],
      }
    },
  },
  plugins: [],
}
