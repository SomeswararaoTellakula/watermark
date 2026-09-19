/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:5055/api/:path*'
      },
      {
        source: '/download/:path*',
        destination: 'http://127.0.0.1:5055/download/:path*'
      },
      {
        source: '/mongo/:path*',
        destination: 'http://127.0.0.1:5055/mongo/:path*'
      }
    ]
  }
}

module.exports = nextConfig
