'use client'
import Link from 'next/link'
import { useEffect, useState } from 'react'
import { getCurrentUser, signOut } from '@/lib/auth'

export function Navbar() {
  const [user, setUser] = useState<{ email: string, name?: string } | null>(null)
  useEffect(() => {
    setUser(getCurrentUser())
  }, [])
  return (
    <header className="sticky top-0 z-50">
      <div className="backdrop-blur-md bg-black/40 border-b border-white/10">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <Link href="/" className="font-semibold text-white">
            <span className="text-neon-purple">Penta</span><span className="text-neon-blue">Mark</span>
          </Link>
          <div className="flex items-center gap-6 text-sm">
            <nav className="flex gap-6">
              <Link className="hover:text-neon-blue transition" href="/embed" prefetch={false}>Embed Watermark</Link>
              <Link className="hover:text-neon-purple transition" href="/verify" prefetch={false}>Verify Image</Link>
              <Link className="hover:text-neon-blue transition" href="/embed-video" prefetch={false}>Embed Video</Link>
              <Link className="hover:text-neon-purple transition" href="/verify-video" prefetch={false}>Verify Video</Link>
              <Link className="hover:text-white/80 transition" href="/dashboard" prefetch={false}>Dashboard</Link>
              <Link className="hover:text-white/80 transition" href="/about">About</Link>
            </nav>
            <div className="flex items-center gap-4">
              {user ? (
                <>
                  <span className="text-white/80">{user.name || user.email}</span>
                  <button
                    className="underline text-white/80 hover:text-white"
                    onClick={() => {
                      signOut()
                      setUser(null)
                    }}
                  >
                    Sign out
                  </button>
                </>
              ) : (
                <>
                  <Link className="underline text-white/80 hover:text-white" href="/login" prefetch={false}>Login</Link>
                  <Link className="underline text-white/80 hover:text-white" href="/signup" prefetch={false}>Sign Up</Link>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
