import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl
  // Always allow assets and APIs
  if (
    pathname.startsWith('/api') ||
    pathname.startsWith('/_next') ||
    pathname.startsWith('/static') ||
    pathname.startsWith('/mongo') ||
    pathname.startsWith('/download') ||
    pathname === '/favicon.ico'
  ) {
    return NextResponse.next()
  }
  const token = req.cookies.get('dm_token')?.value
  const loggedIn = Boolean(token)
  // If user is logged in and visits auth pages, send home
  if (loggedIn && (pathname === '/login' || pathname === '/signup')) {
    const url = req.nextUrl.clone()
    url.pathname = '/'
    return NextResponse.redirect(url)
  }
  // If not logged in, gate protected pages and root
  const protectedPaths = ['/', '/embed', '/verify', '/dashboard', '/embed-video', '/verify-video']
  if (!loggedIn && protectedPaths.includes(pathname)) {
    const url = req.nextUrl.clone()
    // First show signup, then login after successful signup
    url.pathname = '/signup'
    // Remove all query params (including _rsc) to avoid issues with redirected prefetch
    url.search = ''
    return NextResponse.redirect(url)
  }
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!.*\\.).*)'],
}
