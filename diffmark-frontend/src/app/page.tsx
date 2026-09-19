'use client'
import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import Link from 'next/link'

export default function Page() {
  const router = useRouter()
  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('dm_token') : null
    if (!token) {
      router.replace('/signup')
    }
  }, [router])
  return (
    <div className="py-16">
      <section className="text-center space-y-6">
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
          <span className="text-neon-purple">Protect Your Images</span> from <span className="text-neon-blue">Deepfakes</span>
        </h1>
        <p className="text-white/70 max-w-2xl mx-auto">
          AI-powered watermarking that resists generative manipulations. Elegant, fast, and secure.
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/embed"><GradientButton>Start Protecting</GradientButton></Link>
          <Link className="underline text-white/80 hover:text-white" href="/verify">Verify Image</Link>
        </div>
      </section>
      <section className="mt-16 grid md:grid-cols-3 gap-6">
        <GlassCard className="p-6">
          <h3 className="text-xl font-semibold">Deepfake-Resistant</h3>
          <p className="text-white/70 mt-2">Designed to persist through diffusion-based edits and face swaps.</p>
        </GlassCard>
        <GlassCard className="p-6">
          <h3 className="text-xl font-semibold">Fast Embedding</h3>
          <p className="text-white/70 mt-2">Embed robust watermarks in seconds with simple UI.</p>
        </GlassCard>
        <GlassCard className="p-6">
          <h3 className="text-xl font-semibold">Accurate Verification</h3>
          <p className="text-white/70 mt-2">Detect authenticity with confidence scores and bit accuracy.</p>
        </GlassCard>
      </section>
      <section className="mt-20">
        <GlassCard className="p-8">
          <div className="grid md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-1">
              <h2 className="text-2xl font-semibold">How It Works</h2>
            </div>
            <div className="md:col-span-2 grid md:grid-cols-3 gap-4">
              <div className="glass rounded-lg p-4">
                <h4 className="font-medium">Upload</h4>
                <p className="text-white/70">Drag and drop your image.</p>
              </div>
              <div className="glass rounded-lg p-4">
                <h4 className="font-medium">Embed</h4>
                <p className="text-white/70">Insert secret message into robust watermark.</p>
              </div>
              <div className="glass rounded-lg p-4">
                <h4 className="font-medium">Verify</h4>
                <p className="text-white/70">Check authenticity with confidence.</p>
              </div>
            </div>
          </div>
        </GlassCard>
      </section>
    </div>
  )
}
