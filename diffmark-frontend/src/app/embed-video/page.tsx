'use client'
import { useEffect, useRef, useState } from 'react'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import { embedVideo, embedVideoRobust, health } from '@/lib/api'

export default function EmbedVideoPage() {
  const [file, setFile] = useState<File | null>(null)
  const [wm, setWm] = useState('')
  const [loading, setLoading] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [id, setId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [ffmpegReady, setFfmpegReady] = useState<boolean | null>(null)

  const [robust, setRobust] = useState(false)

  useEffect(() => {
    (async () => {
      try {
        const h = await health()
        setFfmpegReady(h.ffmpeg !== 'down')
      } catch {
        setFfmpegReady(null)
      }
    })()
  }, [])

  async function onEmbed() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      if (ffmpegReady === false) {
        throw new Error('Video processing unavailable: ffmpeg not installed on server')
      }
      const res = robust ? await embedVideoRobust(file, wm) : await embedVideo(file, wm)
      setResultUrl(res.watermarked_video_url)
      if (res.id) setId(res.id)
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Failed to embed watermark on video'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">Embed Video Watermark</h1>
      {error && <div className="glass rounded-lg p-4 border border-red-400/40 text-red-300">Error: {error}</div>}
      <GlassCard className="p-6">
        {ffmpegReady === false && (
          <div className="glass rounded-lg p-4 border border-yellow-400/40 text-yellow-200 mb-4">
            ffmpeg is not installed on the server; video embedding will fail until it is available.
          </div>
        )}
        <div
          className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-neon-purple transition"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files?.[0]
            if (f) setFile(f)
          }}
        >
          <p className="text-white/70">Drag & drop a video or select a file</p>
          <input
            id="fileInput"
            type="file"
            accept="video/*"
            className="mt-4 glass rounded-lg px-4 py-3 block w-full"
            ref={inputRef}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <div className="mt-3 text-white/60 text-sm">Or drag-and-drop a video above</div>
        </div>
        <div className="mt-6 flex items-center gap-4">
          <input
            className="glass rounded-lg px-4 py-3 flex-1"
            placeholder="Watermark text"
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
            <video src={URL.createObjectURL(file)} className="rounded-lg w-full h-auto" controls />
          ) : (
            <p className="text-white/60">No video selected</p>
          )}
        </GlassCard>
        <GlassCard className="p-6">
          <h3 className="font-medium mb-3">Watermarked</h3>
          {resultUrl ? (
            <video src={resultUrl} className="rounded-lg w-full h-auto" controls />
          ) : (
            <p className="text-white/60">No result yet</p>
          )}
        </GlassCard>
      </div>
      {resultUrl && (
        <div className="flex items-center justify-end gap-6">
          <a href={resultUrl} download className="underline">Download MP4</a>
          {id && <a href={`/verify-video?id=${id}${robust ? '&robust=1' : ''}`} prefetch-prevent="true" className="underline">Verify This Video</a>}
        </div>
      )}
    </div>
  )
}
