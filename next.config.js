/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';

const nextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  output: 'export',
  basePath: isProd ? '/spendingproofs' : '',
  images: {
    unoptimized: true,
  },
  transpilePackages: ['@rainbow-me/rainbowkit'],
  webpack: (config) => {
    config.resolve.fallback = { fs: false, net: false, tls: false };
    config.externals = [...(config.externals || []), 'pino-pretty', 'encoding'];
    return config;
  },
};

module.exports = nextConfig;
