import { api } from './api'

type User = {
  id: string
  email: string
  name?: string
}

const USER_KEY = 'dm_user'
const TOKEN_KEY = 'dm_token'

export function getCurrentUser(): User | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export async function signUp(name: string, email: string, password: string): Promise<User> {
  const res = await api.post('/auth/signup', { name, email, password })
  const { user } = res.data as { user: User, token: string }
  // Do not persist token on signup; require explicit login
  return user
}

export async function signIn(email: string, password: string): Promise<User> {
  const res = await api.post('/auth/login', { email, password })
  const { user, token } = res.data as { user: User, token: string }
  if (typeof window !== 'undefined') {
    localStorage.setItem(USER_KEY, JSON.stringify(user))
    localStorage.setItem(TOKEN_KEY, token)
    document.cookie = `dm_token=${token}; Path=/; Max-Age=${60 * 60 * 24 * 7}`
  }
  return user
}

export function signOut() {
  if (typeof window === 'undefined') return
  localStorage.removeItem(USER_KEY)
  localStorage.removeItem(TOKEN_KEY)
  document.cookie = 'dm_token=; Path=/; Max-Age=0'
}
