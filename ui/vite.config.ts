import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The UI calls the API under /api/... (same origin, so the browser needs no CORS).
// In development Vite forwards /api to the FastAPI dev server (make dev); in Docker,
// nginx does the same job (ui/nginx.conf).
const API_URL = process.env.API_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API_URL, rewrite: (path) => path.replace(/^\/api/, '') },
    },
  },
  test: {
    environment: 'jsdom', // a simulated browser DOM in Node
    setupFiles: ['./src/test/setup.ts'],
  },
})
