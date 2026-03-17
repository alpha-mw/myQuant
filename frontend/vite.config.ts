import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return
          }

          if (id.includes('react-markdown') || id.includes('/remark-') || id.includes('/mdast-') || id.includes('/micromark')) {
            return 'markdown'
          }

          if (id.includes('lightweight-charts')) {
            return 'lightweight-charts'
          }

          if (id.includes('recharts') || id.includes('/d3-')) {
            return 'recharts'
          }

          if (id.includes('@tanstack/')) {
            return 'tanstack'
          }

          if (
            id.includes('react-router-dom') ||
            id.includes('/react/') ||
            id.includes('/react-dom/') ||
            id.includes('/scheduler/') ||
            id.includes('/react-is/')
          ) {
            return 'react-core'
          }

          if (id.includes('lucide-react')) {
            return 'icons'
          }

          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
