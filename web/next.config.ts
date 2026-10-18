import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static site: every page is built once from the exported JSON in public/data (no server, no database).
  output: "export",
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
