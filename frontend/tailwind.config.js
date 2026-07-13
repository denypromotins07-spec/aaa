/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Cyberpunk Quant Palette
        obsidian: '#0a0a0c',
        'obsidian-light': '#121216',
        'neon-cyan': '#00f5ff',
        'neon-magenta': '#ff00ff',
        'neon-green': '#00ff9d',
        'neon-red': '#ff2a6d',
        'glass-bg': 'rgba(18, 18, 22, 0.7)',
        'glass-border': 'rgba(255, 255, 255, 0.1)',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        'xs': '2px',
        'glass': '12px',
        'glass-lg': '24px',
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 245, 255, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 245, 255, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}
