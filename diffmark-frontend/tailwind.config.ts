import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        neon: {
          purple: '#8A2BE2',
          blue: '#00B4FF'
        }
      },
      boxShadow: {
        glow: '0 0 20px rgba(138,43,226,0.3), 0 0 40px rgba(0,180,255,0.2)'
      },
      backgroundImage: {
        'gradient-hero': 'radial-gradient(ellipse at top left, rgba(138,43,226,0.25), transparent 40%), radial-gradient(ellipse at bottom right, rgba(0,180,255,0.25), transparent 40%)'
      }
    }
  },
  plugins: []
}

export default config
