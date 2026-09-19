import './globals.css'
import { ReactNode } from 'react'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'

export const metadata = {
  title: 'Pentamark – Deepfake Protection System',
  description: 'Protect your images from deepfakes with AI-powered watermarking.'
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <div className="min-h-screen bg-gradient-hero">
          <Navbar />
          {children}
          <Footer />
        </div>
      </body>
    </html>
  )
}
