'use client'
import { Suspense, useMemo, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import { verifyVideoById, verifyVideoFile, verifyVideoRobustById } from '@/lib/api'

export const dynamic = 'force-dynamic'

function VerifyVideoContent() {
  const params = useSearchParams()
  const id = params.get('id') || null
  const robust = params.get('robust') === '1'
  const previewUrl = useMemo(() => (id ? `/mongo/${id}` : null), [id])
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ detected: boolean, accuracy: number, confidence: number, watermark_message?: string, robust?: boolean } | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function onVerify() {
    if (!file && !id) return
    setLoading(true)
    setError(null)
    try {
      const res = id ? (robust ? await verifyVideoRobustById(id) : await verifyVideoById(id)) : await verifyVideoFile(file as File)
      setResult(res)
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Verification failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">Verify Video</h1>
      {error && <div className="glass rounded-lg p-4 border border-red-400/40 text-red-300">Error: {error}</div>}
      <GlassCard className="p-6">
        <div
          className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-neon-blue transition"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files?.[0]
            if (f) setFile(f)
          }}
        >
          <p className="text-white/70">Drag & drop a video or select a file</p>
          {!id && (
            <input
              type="file"
              accept="video/*"
              className="mt-4"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          )}
        </div>
        <div className="mt-6 flex justify-end">
          <GradientButton onClick={onVerify} disabled={(!file && !id) || loading}>
            {loading ? 'Scanning...' : 'Verify Authenticity'}
          </GradientButton>
        </div>
      </GlassCard>
      <div className="grid md:grid-cols-2 gap-6">
        <GlassCard className="p-6">
          <h3 className="font-medium mb-3">Video</h3>
          {previewUrl ? (
            <video src={previewUrl} className="rounded-lg w-full h-auto" controls />
          ) : file ? (
            <video src={URL.createObjectURL(file)} className="rounded-lg w-full h-auto" controls />
          ) : (
            <p className="text-white/60">No video selected</p>
          )}
        </GlassCard>
        {result && (
          <GlassCard className="p-6">
            <h3 className="font-medium mb-3">Result</h3>
            <div className={`rounded-lg p-4 ${result.detected ? 'bg-green-500/10 border border-green-400/40' : 'bg-red-500/10 border border-red-400/40'}`}>
              <div className="text-xl font-semibold">{result.detected ? 'Authentic' : 'Not Authentic'}</div>
              {result.detected && (
                <div className="text-xs mt-1 px-2 py-0.5 rounded-full inline-block bg-white/10 text-white/70 border border-white/10">
                  {result.robust ? 'Invisible Robust Watermark' : 'Visible Standard Watermark'}
                </div>
              )}
              <div className="text-white/80 mt-4 text-sm">Accuracy: {(result.accuracy * 100).toFixed(0)}%</div>
              <div className="text-white/80 text-sm">Confidence: {(result.confidence * 100).toFixed(0)}%</div>
              {result.watermark_message && <div className="text-white mt-2">Watermark: {result.watermark_message}</div>}
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  )
}

export default function VerifyVideoPage() {
  return (
    <Suspense fallback={<div className="py-12"><p className="text-white/70">Loading...</p></div>}>
      <VerifyVideoContent />
    </Suspense>
  )
}
