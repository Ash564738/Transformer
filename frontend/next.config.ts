import type { NextConfig } from "next";

// The frontend calls the Flask backend directly from the browser
// (see NEXT_PUBLIC_BACKEND_URL in src/lib/api.ts) — long-running /predict
// requests on large datasets were getting reset by Next.js's dev-server
// rewrite proxy, and calling Flask directly (with CORS enabled there)
// sidesteps that entirely. No rewrite needed here.
const nextConfig: NextConfig = {};

export default nextConfig;
