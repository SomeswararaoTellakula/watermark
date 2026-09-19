'use client'
import { Suspense, useEffect, useMemo, useState } from 'react'
import { verifyImage, verifyById, verifyRobustById, verifyRobustFile } from '@/lib/api'
import { useSearchParams } from 'next/navigation'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import { ProgressBar } from '@/components/ProgressBar'
import { ResultChart } from '@/components/ResultChart'
import { ResultDonut } from '@/components/ResultDonut'
import { Sparkline } from '@/components/Sparkline'

export const dynamic = 'force-dynamic'

function VerifyContent() {
  const params = useSearchParams()
  const id = params.get('id') || null
  const robust = params.get('robust') === '1'
  const previewUrl = useMemo(() => (id ? `/mongo/${id}` : null), [id])
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ detected: boolean, accuracy: number, confidence: number, watermark_message?: string, watermark_bits?: string, robust?: boolean } | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function onVerify() {
    if (!file && !id) return
    setLoading(true)
    setError(null)
    try {
      // First attempt according to current mode
      let res = id ? (robust ? await verifyRobustById(id) : await verifyById(id)) : (robust ? await verifyRobustFile(file as File) : await verifyImage(file as File))
      // Fallback: if standard verification didn't find a message or detection failed, try robust
      if (!robust && (!res?.detected || !res?.watermark_message)) {
        const robustRes = id ? await verifyRobustById(id) : await verifyRobustFile(file as File)
        // Prefer the robust result if it detects or contains a watermark message
        if (robustRes?.detected || robustRes?.watermark_message) {
          res = robustRes
        }
      }
      setResult(res)
      try {
        if (typeof window !== 'undefined') {
          const raw = localStorage.getItem('verify_metrics') || '[]'
          const arr = JSON.parse(raw) as Array<{ t: number, a: number, c: number }>
          arr.push({ t: Date.now(), a: res.accuracy, c: res.confidence })
          localStorage.setItem('verify_metrics', JSON.stringify(arr.slice(-50)))
        }
      } catch {}
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Verification failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function bitsFromMessage(msg?: string | null, L = 30) {
    if (!msg) return null
    const b = Array.from(msg).map(c => c.charCodeAt(0).toString(2).padStart(8, '0')).join('')
    return (b + '0'.repeat(L)).slice(0, L)
  }

  useEffect(() => {
    if (file && !loading && !id) {
      onVerify()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">Verify Image</h1>
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
          <p className="text-white/70">Drag & drop an image or select a file</p>
          {!id && (
            <input
              type="file"
              accept="image/*"
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
          <h3 className="font-medium mb-3">Image</h3>
          {previewUrl ? (
            <img src={previewUrl} alt="to-verify" className="rounded-lg w-full h-auto" />
          ) : file ? (
            <img src={URL.createObjectURL(file)} alt="to-verify" className="rounded-lg w-full h-auto" />
          ) : (
            <p className="text-white/60">No image selected</p>
          )}
        </GlassCard>
        {result?.watermark_message && (
          <GlassCard className="p-6">
            <h3 className="font-medium mb-3">Watermark Message</h3>
            <p className="text-white">{result.watermark_message}</p>
            <div className="mt-4">
              <p className="text-sm text-white/70 mb-1">Binary Watermark</p>
              <div className="glass rounded-lg px-4 py-3 text-white font-mono break-all">
                {result.watermark_bits || bitsFromMessage(result.watermark_message) || 'N/A'}
              </div>
            </div>
          </GlassCard>
        )}
      </div>
      {result && (
        <GlassCard className="p-6">
          <div className="grid md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-1">
              <div className={`rounded-lg p-4 ${result.detected ? 'bg-green-500/10 border border-green-400/40' : 'bg-red-500/10 border border-red-400/40'}`}>
                <h3 className="text-xl font-semibold">{result.detected ? 'Authentic' : 'Not Authentic'}</h3>
                {result.detected && (
                  <div className="text-xs mt-1 px-2 py-0.5 rounded-full inline-block bg-white/10 text-white/70 border border-white/10">
                    {result.robust ? 'Invisible Robust Watermark' : 'Visible Standard Watermark'}
                  </div>
                )}
              </div>
            </div>
            <div className="md:col-span-2 space-y-4">
              <div>
                <p className="text-sm text-white/70 mb-1">Verification Accuracy</p>
                <ProgressBar value={result.accuracy * 100} />
              </div>
              <div>
                <p className="text-sm text-white/70 mb-1">Confidence</p>
                <ProgressBar value={result.confidence * 100} />
              </div>
              <div className="pt-3">
                <p className="text-sm text-white/70 mb-2">Graph Result</p>
                <ResultChart accuracy={result.accuracy} confidence={result.confidence} />
              </div>
              <div className="grid sm:grid-cols-2 gap-4">
                <ResultDonut value={result.accuracy} label="Accuracy" />
                <ResultDonut value={result.confidence} label="Confidence" color="#00bfff" />
              </div>
              <div>
                <p className="text-sm text-white/70 mb-2">Accuracy Trend (local)</p>
                <Trend />
              </div>
            </div>
          </div>
        </GlassCard>
      )}
    </div>
  )
}

function Trend() {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem('verify_metrics') || '[]'
    const arr = JSON.parse(raw) as Array<{ t: number, a: number }>
    const points = arr.map(x => x.a)
    if (points.length === 0) return <p className="text-white/60">No local trend yet.</p>
    return <Sparkline points={points} />
  } catch {
    return null
  }
}

export default function VerifyPage() {
  return (
    <Suspense fallback={<div className="py-12"><p className="text-white/70">Loading...</p></div>}>
      <VerifyContent />
    </Suspense>
  )
}
