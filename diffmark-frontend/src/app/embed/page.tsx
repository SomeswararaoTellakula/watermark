'use client'
import { useRef, useState } from 'react'
import { embedWatermark, embedWatermarkRobust, robustStatus, robustReload } from '@/lib/api'
import Link from 'next/link'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'

export default function EmbedPage() {
  const [file, setFile] = useState<File | null>(null)
  const [wm, setWm] = useState('')
  const [loading, setLoading] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [id, setId] = useState<string | null>(null)
  const [robust, setRobust] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  async function onEmbed() {
    if (!file) return
    setLoading(true)
    setError(null)
    setNotice(null)
    try {
      if (robust) {
        try {
          await robustReload()
        } catch {}
      }
      const res = robust ? await embedWatermarkRobust(file, wm) : await embedWatermark(file, wm)
      setResultUrl(res.watermarked_image_url)
      if (res.id) setId(res.id)
      if (res as any && (res as any).robust) {
        setNotice('Invisible watermark embedded (metadata only)')
      } else {
        setNotice('Visible overlay embedded')
      }
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Failed to embed watermark'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">Embed Watermark</h1>
      {error && <div className="glass rounded-lg p-4 border border-red-400/40 text-red-300">Error: {error}</div>}
      {notice && <div className="glass rounded-lg p-4 border border-green-400/40 text-green-300">{notice}</div>}
      <GlassCard className="p-6">
        <div
          className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-neon-purple transition"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files?.[0]
            if (f) setFile(f)
          }}
        >
          <p className="text-white/70">Drag & drop an image or select a file</p>
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            className="mt-4 glass rounded-lg px-4 py-3 block w-full"
            ref={inputRef}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <div className="mt-3 text-white/60 text-sm">Or drag-and-drop an image above</div>
        </div>
        <div className="mt-6 flex items-center gap-4">
          <input
            className="glass rounded-lg px-4 py-3 flex-1"
            placeholder="Secret watermark text"
            value={wm}
            onChange={(e) => setWm(e.target.value)}
          />
          <label className="flex items-center gap-2 text-white/80">
            <input type="checkbox" checked={robust} onChange={(e) => setRobust(e.target.checked)} />
            Robust (Diffusion)
          </label>
          <GradientButton onClick={onEmbed} disabled={!file || loading}>
            {loading ? 'Embedding...' : 'Embed Watermark'}
          </GradientButton>
        </div>
      </GlassCard>
      <div className="grid md:grid-cols-2 gap-6">
        <GlassCard className="p-6">
          <h3 className="font-medium mb-3">Original</h3>
          {file ? (
            <img src={URL.createObjectURL(file)} alt="original" className="rounded-lg w-full h-auto" />
          ) : (
            <p className="text-white/60">No image selected</p>
          )}
        </GlassCard>
        <GlassCard className="p-6">
          <h3 className="font-medium mb-3">Watermarked</h3>
          {resultUrl ? (
            <img src={resultUrl} alt="watermarked" className="rounded-lg w-full h-auto" />
          ) : (
            <p className="text-white/60">No result yet</p>
          )}
        </GlassCard>
      </div>
      {resultUrl && (
        <div className="flex items-center justify-end gap-6">
          <a href={resultUrl} download className="underline">Download PNG</a>
          {id && (
                <Link href={`/verify?id=${id}${robust ? '&robust=1' : ''}`} prefetch={false} className="underline">
              Verify This Image
            </Link>
          )}
        </div>
      )}
    </div>
  )
}
