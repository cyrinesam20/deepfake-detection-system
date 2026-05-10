export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          400: "#22d3ee", 500: "#06b6d4", 600: "#0891b2"
        },
        dark: {
          600: "#1e293b", 700: "#0f172a", 800: "#0d1117",
          950: "#080b12"
        },
        success: { 400: "#4ade80" },
        danger: { 400: "#f87171", 500: "#ef4444" },
      },
      fontFamily: {
        display: ["Outfit", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      }
    }
  },
  plugins: []
}