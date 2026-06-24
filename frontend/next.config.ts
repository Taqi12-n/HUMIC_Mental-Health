import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow cross-origin requests in dev for audio playback
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
        ],
      },
    ];
  },
};

export default nextConfig;
