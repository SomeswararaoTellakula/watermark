'use client'
import { useEffect, useState } from 'react'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import { getHistory } from '@/lib/api'

type Item = { id: string, filename?: string, watermarked_image_url: string, wm_text?: string, robust?: boolean, uploadDate?: string }

export default function DashboardPage() {
  const [items, setItems] = useState<Item[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const res = await getHistory()
        setItems(res.items || [])
      } catch (e: any) {
        setError('Failed to load history')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">Your Dashboard</h1>
      {error && <div className="glass rounded-lg p-4 border border-red-400/40 text-red-300">Error: {error}</div>}
      <GlassCard className="p-6">
        {loading ? (
          <p className="text-white/70">Loading...</p>
        ) : items.length === 0 ? (
          <p className="text-white/70">No history yet. Embed a watermark to see it here.</p>
        ) : (
          <div className="grid md:grid-cols-3 gap-6">
            {items.map((it) => (
              <div key={it.id} className="glass rounded-lg p-4 space-y-3">
                <img src={it.watermarked_image_url} alt={it.filename || it.id} className="rounded-lg w-full h-auto" />
                <div className="flex items-center justify-between text-sm">
                  <div className="text-white/80">
                    <div className="font-medium">{it.wm_text || 'Pentamark'}</div>
                    <div className="text-white/60">{it.robust ? 'Robust' : 'Standard'}</div>
                  </div>
                  <a href={it.watermarked_image_url} download className="underline hover:text-white text-white/80">
                    Download
                  </a>
                </div>
                {it.uploadDate && <div className="text-xs text-white/50">{new Date(it.uploadDate).toLocaleString()}</div>}
              </div>
            ))}
          </div>
        )}
      </GlassCard>
      <div className="flex justify-end">
        <a href="/embed"><GradientButton>Embed New</GradientButton></a>
      </div>
    </div>
  )
}
