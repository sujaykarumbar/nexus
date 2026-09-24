/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        nexus: {
          900: '#070B14',
          850: '#0B1120',
          800: '#0F172A',
          700: '#1E293B',
          600: '#334155',
          accent: '#38BDF8',
          purple: '#818CF8',
          emerald: '#34D399',
          amber: '#FBBF24',
          rose: '#FB7185'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'glow': '0 0 20px -5px rgba(56, 189, 248, 0.3)',
        'glow-purple': '0 0 20px -5px rgba(129, 140, 248, 0.3)',
      }
    },
  },
  plugins: [],
}
