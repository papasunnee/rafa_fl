import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Backend (ui/backend) is expected on :8000 in local dev.
      '/decide': 'http://localhost:8000',
      '/run': 'http://localhost:8000',
    },
  },
})
