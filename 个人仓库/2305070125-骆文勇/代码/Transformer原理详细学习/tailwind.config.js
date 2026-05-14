/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        monokai: {
          bg: '#272822',
          fg: '#F8F8F2',
          comment: '#75715E',
          red: '#F92672',
          orange: '#FD971F',
          yellow: '#E6DB74',
          green: '#A6E22E',
          blue: '#66D9EF',
          purple: '#AE81FF',
          pink: '#F92672',
          brown: '#75715E',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}