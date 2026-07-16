import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Backend (ui/backend) is expected on :8000 in local dev.
      // '/run' as a plain string is a PREFIX match, so it would also
      // wrongly intercept the frontend's own /runner route (and forward it
      // to the backend, which has no such path -> 404). Keys starting with
      // '^' are regex-matched instead, so this matches only /run and
      // /run/<job_id>, not /runner.
      '/decide': 'http://localhost:8000',
      '^/run($|/)': 'http://localhost:8000',
    },
  },
})
