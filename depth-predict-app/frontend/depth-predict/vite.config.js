import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const repoName = 'DepthPredict';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production' ? `/${repoName}/` : '/',
})
