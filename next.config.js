/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  output: 'export',
  basePath: '/spendingproofs',
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
