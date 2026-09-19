'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { GlassCard } from '@/components/GlassCard'
import { GradientButton } from '@/components/GradientButton'
import { signIn } from '@/lib/auth'

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await signIn(email, password)
      router.push('/')
    } catch (e: any) {
      const msg = e?.response?.data?.error || 'Login failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="py-12 space-y-8">
      <h1 className="text-3xl font-semibold text-center">Login</h1>
      {error && <div className="glass rounded-lg p-4 border border-red-400/40 text-red-300">Error: {error}</div>}
      <GlassCard className="p-6 max-w-lg mx-auto">
        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-white/80 mb-1">Email</label>
            <input
              type="email"
              required
              className="glass rounded-lg px-4 py-3 w-full"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm text-white/80 mb-1">Password</label>
            <input
              type="password"
              required
              className="glass rounded-lg px-4 py-3 w-full"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="flex items-center justify-between">
            <a href="/signup" className="underline text-white/80 hover:text-white">Create an account</a>
            <GradientButton type="submit" disabled={loading}>
              {loading ? 'Logging in...' : 'Login'}
            </GradientButton>
          </div>
        </form>
      </GlassCard>
    </div>
  )
}
