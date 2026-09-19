import { GlassCard } from '@/components/GlassCard'

export default function AboutPage() {
  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold">About Pentamark</h1>
      <GlassCard className="p-6">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-xl font-semibold">Diffusion Models</h3>
            <p className="text-white/70 mt-2">
              Diffusion models iteratively refine noise into coherent images using learned denoising steps.
              Watermarking that is resistant to these edits requires robust embedding in semantic features.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold">Deepfake Problem</h3>
            <p className="text-white/70 mt-2">
              Synthetic media can alter identity and content. A persistent watermark helps establish provenance
              and authenticity, even after manipulations like face swaps or style transfers.
            </p>
          </div>
        </div>
      </GlassCard>
      <div className="grid md:grid-cols-3 gap-6">
        <GlassCard className="p-6 text-center">
          <p className="text-4xl font-bold">95%</p>
          <p className="text-white/70">Average bit accuracy</p>
        </GlassCard>
        <GlassCard className="p-6 text-center">
          <p className="text-4xl font-bold">0.3s</p>
          <p className="text-white/70">Avg. embed time</p>
        </GlassCard>
        <GlassCard className="p-6 text-center">
          <p className="text-4xl font-bold">99%</p>
          <p className="text-white/70">Verification success</p>
        </GlassCard>
      </div>
    </div>
  )
}
