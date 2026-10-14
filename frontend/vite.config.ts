import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // The 3D view (three.js, ~1.1 MB, ~300 kB gzipped) is its own lazily loaded chunk: the
    // dashboard (~90 kB gzipped) appears first and the heart streams in after it.
    chunkSizeWarningLimit: 1200,
  },
})
