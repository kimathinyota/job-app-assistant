// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Use the .dark class on html tag
  theme: {
    extend: {
      colors: {
        // RoleCraft Brand Palette
        brand: {
          primary: '#0F172A', // Deep Navy (slate-900)
          accent: '#3B82F6',  // Electric Blue (blue-500)
          'accent-hover': '#2563EB', // Blue-600
        },
        bg: {
          light: '#F8FAFC', // Ice White (slate-50)
          dark: '#020617',  // Void Navy (slate-950)
        },
        surface: {
          light: '#FFFFFF',
          dark: '#0F172A',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'executive': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.06)',
      }
    },
  },
  plugins: [],
}